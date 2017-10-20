import numpy
np = numpy

import theano
import theano.tensor as T

from lasagne.layers.base import Layer
from lasagne.random import get_rng

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class DropoutLayer(Layer):
    """
    mask --> self.mask, so that we can access it and fix it (with "givens")
    """
    def __init__(self, incoming, p=0.5, rescale=True, shared_axes=(),
                 **kwargs):
        super(DropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale
        self.shared_axes = tuple(shared_axes)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic or self.p == 0:
            return input
        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1)

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            # use nonsymbolic shape for dropout mask if possible
            mask_shape = self.input_shape
            if any(s is None for s in mask_shape):
                mask_shape = input.shape

            # apply dropout, respecting shared axes
            if self.shared_axes:
                shared_axes = tuple(a if a >= 0 else a + input.ndim
                                    for a in self.shared_axes)
                mask_shape = tuple(1 if a in shared_axes else s
                                   for a, s in enumerate(mask_shape))
            self.mask = self._srng.binomial(mask_shape, p=retain_prob,
                                       dtype=input.dtype)
            if self.shared_axes:
                bcast = tuple(bool(s == 1) for s in mask_shape)
                self.mask = T.patternbroadcast(self.mask, bcast)

            self.mask_shape = mask_shape

            return input * self.mask



# TODO:
import lasagne
import theano
import theano.tensor as T
from lasagne.random import set_rng
from theano.tensor.shared_randomstreams import RandomStreams

lrdefault = 1e-3    
    
class MCdropout_MLP(object):

    def __init__(self,n_hiddens,n_units, output_type='categorical', n_inputs=784, n_outputs=3, drop_prob=.5):
        self.__dict__.update(locals())
        
        layer = lasagne.layers.InputLayer([None,n_inputs])
        

        self.weight_shapes = list()        
        self.weight_shapes.append((n_inputs,n_units))
        for i in range(1,n_hiddens):
            self.weight_shapes.append((n_units,n_units))
        self.weight_shapes.append((n_units,n_outputs))
        self.num_params = sum(ws[1] for ws in self.weight_shapes)
        
        self.layers = []
        for j,ws in enumerate(self.weight_shapes):
            layer = lasagne.layers.DenseLayer(
                layer,ws[1],
                nonlinearity=lasagne.nonlinearities.rectify
            )
            self.layers.append(layer)
            if j!=len(self.weight_shapes)-1:
                layer = DropoutLayer(layer, p=self.drop_prob)
                self.layers.append(layer)
        
        if output_type == 'categorical':
            layer.nonlinearity = lasagne.nonlinearities.softmax
        elif output_type == 'real':
            layer.nonlinearity = lasagne.nonlinearities.linear
        else:
            assert False

        self.input_var = T.matrix('input_var')
        self.target_var = T.matrix('target_var')
        self.learning_rate = T.scalar('leanring_rate')
        
        self.layer = layer
        self.y = lasagne.layers.get_output(layer,self.input_var)
        self.y_det = lasagne.layers.get_output(layer,self.input_var,
                                               deterministic=True)

        # getting the dropout masks: we need to wait until after calling get_output on the layers, so their masks are defined
        self.masks = []
        self.mask_sizes = []
        #self.drop_prob = .5 # TODO: hard coded
        for layer in self.layers:
            if type(layer) is DropoutLayer:
                self.masks.append( layer.mask ) # for fixed mask dropout
                print layer.input_shape
                # TODO: very very hacky (e.g. for MLP only!)
                self.mask_sizes.append((1,layer.input_shape[1]))
        
        if output_type == 'categorical':
            losses = lasagne.objectives.categorical_crossentropy(self.y, self.target_var)
        elif output_type == 'real':
            losses = lasagne.objectives.squared_error(self.y, self.target_var)
        else:
            assert False 

        self.loss = losses.mean()
        self.params = lasagne.layers.get_all_params(self.layer)
        self.updates = lasagne.updates.adam(self.loss,self.params,
                                            self.learning_rate)

        print '\tgetting train_func'
        self.train_func_ = theano.function([self.input_var,
                                            self.target_var,
                                            self.learning_rate],
                                           self.loss,
                                           updates=self.updates)
        
        print '\tgetting useful_funcs'
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y_det.argmax(1))
        self.predict_fixed_mask = theano.function([self.input_var] + self.masks,self.y)
        
    def sample_qyx(self):
        """ return a function that will make predictions with a fixed random mask"""
        masks = [np.random.binomial(1,1-self.drop_prob, mask_size).astype('float32') for mask_size in self.mask_sizes]
        return lambda x : self.predict_fixed_mask(x, *masks)

    def train_func(self,x,y,n,lr=lrdefault,w=1.0):
        return self.train_func_(x,y,lr)

    def save(self,save_path,notes=[]):
        np.save(save_path, [p.get_value() for p in self.params]+notes)

    def load(self,save_path):
        values = np.load(save_path)
        notes = values[-1]
        values = values[:-1]

        if len(self.params) != len(values):
            raise ValueError("mismatch: got %d values to set %d parameters" %
                             (len(values), len(self.params)))

        for p, v in zip(self.params, values):
            if p.get_value().shape != v.shape:
                raise ValueError("mismatch: parameter has shape %r but value to "
                                 "set has shape %r" %
                                 (p.get_value().shape, v.shape))
            else:
                p.set_value(v)

        return notes

    
