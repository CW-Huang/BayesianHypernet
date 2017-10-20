# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:32:10 2017

@author: Chin-Wei

MADE adapted from 
https://github.com/mgermain/MADE

` MADE: Masked Autoencoder for Distribution Estimation `

"""


import copy
import theano
import theano.tensor as T
import numpy as np

# Limited but works on GPU
from theano.sandbox.rng_mrg import MRG_RandomStreams  
from theano.tensor.shared_randomstreams import RandomStreams
floatX = theano.config.floatX
linear = lambda x: x

class MaskGenerator(object):

    def __init__(self, input_size, hidden_sizes, l, random_seed=1234):
        self._random_seed = random_seed
        self._mrng = MRG_RandomStreams(seed=random_seed)
        self._rng = RandomStreams(seed=random_seed)

        self._hidden_sizes = hidden_sizes
        self._input_size = input_size
        self._l = l

        self.ordering = theano.shared(np.arange(input_size, 
                                                dtype=theano.config.floatX), 
                                      'ordering', 
                                      borrow=False)

        # Initial layer connectivity
        self.layers_connectivity = [theano.shared((self.ordering + 1).eval(), 
                                                  'layer_connectivity_input', 
                                                  borrow=False)]
        for i in range(len(self._hidden_sizes)):
            lc = theano.shared(np.zeros((self._hidden_sizes[i]),dtype=floatX), 
                               'layer_connectivity_hidden{0}'.format(i),
                               borrow=False)
            self.layers_connectivity += [lc]
        self.layers_connectivity += [self.ordering]

        ## Theano functions
        new_ordering = self._rng.shuffle_row_elements(self.ordering)
        updates = [(self.ordering, new_ordering), 
                   (self.layers_connectivity[0], new_ordering + 1)]
        self.shuffle_ordering = theano.function(name='shuffle_ordering',
                                                inputs=[],
                                                updates=updates)

        self.layers_connectivity_updates = []
        for i in range(len(self._hidden_sizes)):
            lcu = self._get_hidden_layer_connectivity(i)
            self.layers_connectivity_updates += [lcu]
        
        hsizes = range(len(self._hidden_sizes))
        updates = [(self.layers_connectivity[i+1], 
                    self.layers_connectivity_updates[i]) for i in hsizes]
        self.sample_connectivity = theano.function(name='sample_connectivity',
                                                   inputs=[],
                                                   updates=updates)

        # Save random initial state
        self._initial_mrng_rstate = copy.deepcopy(self._mrng.rstate)
        self._initial_mrng_state_updates = [sup[0].get_value() for sup in 
                                            self._mrng.state_updates]

        # Ensuring valid initial connectivity
        self.sample_connectivity()

    def reset(self):
        # Set Original ordering
        self.ordering.set_value(np.arange(self._input_size, 
                                          dtype=theano.config.floatX))

        # Reset RandomStreams
        self._rng.seed(self._random_seed)

        # Initial layer connectivity
        self.layers_connectivity[0].set_value((self.ordering + 1).eval())
        for i in range(1, len(self.layers_connectivity)-1):
            value = np.zeros((self._hidden_sizes[i-1]), 
                             dtype=theano.config.floatX)
            self.layers_connectivity[i].set_value(value)
        self.layers_connectivity[-1].set_value(self.ordering.get_value())

        # Reset MRG_RandomStreams (GPU)
        self._mrng.rstate = self._initial_mrng_rstate
        states_values = zip(self._mrng.state_updates, 
                            self._initial_mrng_state_updates)
        for state, value in states_values:
            state[0].set_value(value)

        self.sample_connectivity()

    def _get_p(self, start_choice):
        start_choice_idx = (start_choice-1).astype('int32')
        prob = T.nnet.nnet.softmax(self._l * T.arange(start_choice, 
                                                      self._input_size, 
                                                      dtype=floatX))[0]
        p_vals = T.concatenate([T.zeros((start_choice_idx,)),prob])
        p_vals = T.inc_subtensor(p_vals[start_choice_idx], 1.)  
        return p_vals

    def _get_hidden_layer_connectivity(self, layerIdx):
        layer_size = self._hidden_sizes[layerIdx]
        if layerIdx == 0:
            lc = self.layers_connectivity[layerIdx]
            p_vals = self._get_p(T.min(lc))
        else:
            lc = self.layers_connectivity_updates[layerIdx-1]
            p_vals = self._get_p(T.min(lc))

        return T.sum(
            T.cumsum(self._mrng.multinomial(
            pvals=T.tile(p_vals[::-1][None, :],(layer_size, 1)), 
            dtype=floatX), axis=1), axis=1
        )

    def _get_mask(self, layerIdxIn, layerIdxOut):
        return (self.layers_connectivity[layerIdxIn][:, None] <= 
                self.layers_connectivity[layerIdxOut][None, :]).astype(floatX)

    def get_mask_layer_UPDATE(self, layerIdx):
        return self._get_mask(layerIdx, layerIdx + 1)

    def get_direct_input_mask_layer_UPDATE(self, layerIdx):
        return self._get_mask(0, layerIdx)

    def get_direct_output_mask_layer_UPDATE(self, layerIdx):
        return self._get_mask(layerIdx, -1)





class Layer(object):

    def __init__(self, layerIdx, input, n_in, n_out, 
                 weights_initialization, activation=None):
        self.input = input
        self._activation = activation
        self.layerIdx = layerIdx
        self.n_in = n_in
        self.n_out = n_out
        self.weights_initialization = weights_initialization

        # Init weights and biases
        self.W = theano.shared(weights_initialization((n_in, n_out)), 
                               'W{}'.format(layerIdx), borrow=True)
        self.b = theano.shared(np.zeros((n_out,), dtype=floatX), 
                               'b{}'.format(layerIdx), borrow=True)

        # Output
        self.lin_output = T.dot(input, self.W) + self.b

        # Parameters of the layer
        self.params = [self.W, self.b]

    def _output(self):
        return (self.lin_output if self._activation is None 
                                else self._activation(self.lin_output))

    @property
    def output(self):
        return self._output()


class MaskedLayer(Layer):

    def __init__(self, mask_generator, **kargs):
        Layer.__init__(self, **kargs)

        self.mask_generator = mask_generator
        maskname = 'weights_mask{}'.format(self.layerIdx)
        self.weights_mask = theano.shared(np.ones((self.n_in, self.n_out), 
                                                  dtype=floatX), 
                                          maskname, 
                                          borrow=True)

        # Output
        self.lin_output = T.dot(self.input, 
                                self.W * self.weights_mask) + self.b

        mupdate = mask_generator.get_mask_layer_UPDATE(self.layerIdx)
        self.shuffle_update = [(self.weights_mask,mupdate)]


class ConditionningMaskedLayer(MaskedLayer):

    def __init__(self, use_cond_mask=False, **kargs):
        MaskedLayer.__init__(self, **kargs)

        if use_cond_mask:
            self.U = theano.shared(self.weights_initialization((self.n_in, 
                                                                self.n_out)), 
                                   name='U{}'.format(self.layerIdx), 
                                   borrow=True)

            # Output
            self.lin_output += T.dot(T.ones_like(self.input), 
                                     self.U * self.weights_mask)

            self.params += [self.U]
CondMask = ConditionningMaskedLayer

class DirectInputConnectConditionningMaskedLayer(CondMask):

    def __init__(self, direct_input, **kargs):
        ConditionningMaskedLayer.__init__(self, **kargs)

        if direct_input is not None:
            value = np.ones((self.mask_generator._input_size, 
                             self.n_out), dtype=theano.config.floatX)
            name = 'direct_input_weights_mask{}'.format(self.layerIdx)
            self.dinmask = theano.shared(value, name, 
                                                           borrow=True)
            size = (self.mask_generator._input_size, self.n_out)
            value = self.weights_initialization(size)
            name = 'D{}'.format(self.layerIdx)
            self.D = theano.shared(value, name, borrow=True)

            # Output
            self.lin_output += T.dot(direct_input, 
                                     self.D * self.dinmask)

            self.params += [self.D]
            
            getupdate = self.mask_generator.get_direct_input_mask_layer_UPDATE
            self.shuffle_update += [(self.dinmask, 
                                     getupdate(self.layerIdx + 1))]
dinConnCondMask = DirectInputConnectConditionningMaskedLayer

class DirectOutputInputConnectConditionningMaskedOutputLayer(dinConnCondMask):

    def __init__(self, direct_outputs=[], **kargs):
        dinConnCondMask.__init__(self, **kargs)

        self.direct_ouputs_masks = []
        for direct_out_layerIdx, n_direct_out, direct_output in direct_outputs:
            
            value = np.ones((n_direct_out, self.n_out), dtype=floatX)
            name='direct_output_weights_mask{}'.format(self.layerIdx)
            doutmask = theano.shared(value, name, borrow=True)
            
            value=self.weights_initialization((n_direct_out, self.n_out))
            name='direct_ouput_weight{}'.format(self.layerIdx)
            direct_ouput_weight = theano.shared(value, name, 
                                                borrow=True)

            # Output
            self.lin_output += T.dot(direct_output, 
                                     direct_ouput_weight * doutmask)

            self.direct_ouputs_masks += [doutmask]
            self.params += [direct_ouput_weight]
            
            getupdate = self.mask_generator.get_direct_output_mask_layer_UPDATE
            self.shuffle_update += [(doutmask, getupdate(direct_out_layerIdx))]
doutConnCondMask = DirectOutputInputConnectConditionningMaskedOutputLayer



class WInit(object):

    def __init__(self, random_seed):
        self.rng = np.random.mtrand.RandomState(random_seed)

    def _init_range(self, dim):
        return np.sqrt(6. / (dim[0] + dim[1]))

    def Uniform(self, dim):
        init_range = self._init_range(dim)
        return np.asarray(self.rng.uniform(low=-init_range, 
                                           high=init_range, size=dim), 
                                           dtype=floatX)

    def Zeros(self, dim):
        return np.zeros(dim, dtype=theano.config.floatX)

    def Diagonal(self, dim):
        W_values = self.Zeros(dim)
        np.fill_diagonal(W_values, 1)
        return W_values

    def Orthogonal(self, dim):
        max_dim = max(dim)
        return np.linalg.svd(self.Uniform((max_dim, 
                                           max_dim)))[2][:dim[0], :dim[1]]

    def Gaussian(self, dim):
        return np.asarray(self.rng.normal(loc=0, 
                                          scale=self._init_range(dim), 
                                          size=dim), dtype=floatX)


class MADE(object):

    def __init__(self, input,input_size,
                 hidden_sizes=[500],
                 random_seed=1234,
                 hidden_activation=T.nnet.sigmoid,
                 use_cond_mask=False,
                 direct_input_connect="None",
                 direct_output_connect=False,
                 weights_initialization="Uniform",
                 mask_distribution=0,
                 num_output_funcs=1):

        self.shuffled_once = False

        class SeedGenerator(object):
            def __init__(self, random_seed):
                self.rng = np.random.mtrand.RandomState(random_seed)

            def get(self):
                return self.rng.randint(42424242)
                
        self.seed_generator = SeedGenerator(random_seed)

        self.trng = RandomStreams(self.seed_generator.get())

        # Get the weights initializer by string name
        winit = getattr(WInit(self.seed_generator.get()), 
                        weights_initialization)  

        # Initialize the mask
        self.mask_generator = MaskGenerator(input_size, 
                                            hidden_sizes, 
                                            mask_distribution, 
                                            self.seed_generator.get())

        # Initialize layers
        input_layer = CondMask(layerIdx=0,
                               input=input,
                               n_in=input_size,
                               n_out=hidden_sizes[0],
                               activation=hidden_activation,
                               weights_initialization=winit,
                               mask_generator=self.mask_generator,
                               use_cond_mask=use_cond_mask)
        self.layers = [input_layer]
        # Now the hidden layers
        for i in range(1, len(hidden_sizes)):
            previous_layer = self.layers[i - 1]
            output = previous_layer.output
            if direct_input_connect == "Full" and output != input:
                direct_input = input  
            else:
                direct_input = None
            hidden_layer = dinConnCondMask(layerIdx=i,
                                           input=output,
                                           n_in=hidden_sizes[i - 1],
                                           n_out=hidden_sizes[i],
                                           activation=hidden_activation,
                                           weights_initialization=winit,
                                           mask_generator=self.mask_generator,
                                           use_cond_mask=use_cond_mask,
                                           direct_input=direct_input)
            self.layers += [hidden_layer]
        # And the output layer
        outputLayerIdx = len(self.layers)
        previous_layer = self.layers[outputLayerIdx - 1]
        output = previous_layer.output
        if direct_input_connect == "Full" and output != input:
            direct_input = input  
        elif direct_input_connect == "Output" and output != input:
            direct_input = input  
        else:
            direct_input = None
        
        direct_outputs = list()
        if direct_output_connect:            
            for layerIdx, layer in enumerate(self.layers[1:-1]):
                direct_outputs.append(layer.layer_idx,layer.n_in,layer.input)
        
        
        # outputs
        output_layers = list()

        output_layer = doutConnCondMask(layerIdx=outputLayerIdx,
                                        input=previous_layer.output,
                                        n_in=hidden_sizes[outputLayerIdx-1],
                                        n_out=input_size,
                                        activation=linear,
                                        weights_initialization=winit,
                                        mask_generator=self.mask_generator,
                                        use_cond_mask=use_cond_mask,
                                        direct_input=direct_input,
                                        direct_outputs=direct_outputs)
        
        self.layers += [output_layer]
        output_layers.append(output_layer)
                
        # generate fake masks to be replaced
        pseudo_seed = SeedGenerator(random_seed)
        pseudo_generator = MaskGenerator(hidden_sizes[-1], 
                                         [input_size], 
                                         mask_distribution, 
                                         pseudo_seed.get())
        for i in range(num_output_funcs-1):
            
            output_layer = doutConnCondMask(layerIdx=1,
                                            input=previous_layer.output,
                                            n_in=hidden_sizes[outputLayerIdx-1],
                                            n_out=input_size,
                                            activation=linear,
                                            weights_initialization=winit,
                                            mask_generator=pseudo_generator,
                                            use_cond_mask=use_cond_mask,
                                            direct_input=direct_input,
                                            direct_outputs=direct_outputs)
            
            output_layers.append(output_layer)
            
        
        
        
        self.masks = [ll.weights_mask for ll in self.layers] + \
                     [ll.weights_mask for ll in output_layers[1:]]
        self.output_layers = output_layers
        
        
        masks_updates = [upd for layer in self.layers 
                             for upd in layer.shuffle_update]
        self.update_masks = theano.function([],[],updates=masks_updates)
        
        params = [ll.params for ll in self.layers] + \
                 [ll.params for ll in output_layers[1:]]
        self.params = np.concatenate(params).tolist()
        
    def shuffle(self, shuffling_type):
        if shuffling_type == "Once" and self.shuffled_once is False:
            self.mask_generator.shuffle_ordering()
            self.mask_generator.sample_connectivity()
            self.update_masks()
            self.shuffled_once = True
            
            output1 = self.output_layers[0]
            mask_value = output1.weights_mask.get_value()
            for out in self.output_layers[1:]:
                out.weights_mask.set_value(mask_value)
            
            return

        if shuffling_type in ["Ordering", "Full"]:
            self.mask_generator.shuffle_ordering()
        if shuffling_type in ["Connectivity", "Full"]:
            self.mask_generator.sample_connectivity()
        self.update_masks()

    def get_params(self):
        return self.params
