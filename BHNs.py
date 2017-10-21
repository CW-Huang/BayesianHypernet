#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:58:58 2017

@author: Chin-Wei
"""

# TODO: we should have a function for the core hypernet architecture (agnostic of whether we do WN/CNN/full Hnet)

from modules import LinearFlowLayer, IndexLayer, PermuteLayer, SplitLayer, ReverseLayer
from modules import CoupledDenseLayer, ConvexBiasLayer, CoupledWNDenseLayer, \
                    stochasticDenseLayer2, stochasticConv2DLayer, \
                    stochastic_weight_norm
from modules import *
from utils import log_normal
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
RSSV = T.shared_randomstreams.RandomStateSharedVariable
floatX = theano.config.floatX

import lasagne
from lasagne import nonlinearities
rectify = nonlinearities.rectify
softmax = nonlinearities.softmax
from lasagne.layers import get_output
from lasagne.objectives import categorical_crossentropy as cc
from lasagne.objectives import squared_error as se
import numpy as np

from helpers import flatten_list
from helpers import SaveLoadMIXIN


lrdefault = 1e-3
class Base_BHN(object):
    """
    def _get_theano_variables(self):
    def _get_hyper_net(self):
    def _get_primary_net(self):
    def _get_params(self):
    def _get_elbo(self):
    def _get_grads(self):
    def _get_train_func(self):
    def _get_useful_funcs(self):
    """
    
    max_norm = 10
    clip_grad = 5
    
    def __init__(self,
                flow='RealNVP',
                #flow_depth=4, # TODO: for now, we just keep using the "coupling" argument!
                 lbda=1.,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 opt='adam',
                 prior = log_normal,
                 output_type = 'categorical',
                 init_batch = None):
        
        self.__dict__.update(locals())
        
        self._get_theano_variables()
        
        if perdatapoint:
            self.wd1 = self.input_var.shape[0]
        else:
            self.wd1 = 1
    
        
        print('\tbuilding hyper net')
        self._get_hyper_net()
        print('\tbuilding primary net')
        self._get_primary_net()
        print('\tgetting params')
        self._get_params()
        print('\tgetting elbo')
        self._get_elbo()
        print('\tgetting grads')
        self._get_grads()
        print('\tgetting train funcs')
        self._get_train_func()
        print('\tgetting useful funcs')
        self._get_useful_funcs()
        
        
        params0 = lasagne.layers.get_all_param_values([self.h_net,self.p_net])
        params = lasagne.layers.get_all_params([self.h_net,self.p_net])
        updates = {p:p0 for p, p0 in zip(params,params0)}
        self.reset = theano.function([],None,
                                      updates=updates)
        #self.add_reset('init')
        
        
        if init_batch is not None:
            print('\tre-init primary net')
            self._init_pnet(init_batch)
    
    def _get_theano_variables(self):
        self.input_var = T.matrix('input_var')
        self.target_var = T.matrix('target_var')
        self.dataset_size = T.scalar('dataset_size')
        self.learning_rate = T.scalar('learning_rate')
        # TODO: fix name
        self.weight = T.scalar('weight')
        
    def _get_hyper_net(self):
        """
        hypernet outputing weight parameters of the primary net.
        structure to be specified.
        
        DEFINE h_net, weights, logdets
        """
        raise NotImplementedError("BaseBayesianHypernet does not implement"
                                  "the _get_hyper_net() method")

    
    def _get_primary_net(self):
        """
        main structure of the predictive network (to be specified).
        
        DEFINE p_net, y
        """
        raise NotImplementedError("BaseBayesianHypernet does not implement"
                                  "the _get_primary_net() method")

    def _get_params(self):
        
        params = lasagne.layers.get_all_params([self.h_net,self.p_net])
        self.params = list()
        for param in params:
            if type(param) is not RSSV:
                self.params.append(param)
    
    def _get_elbo(self):
        """
        negative elbo, an upper bound on NLL
        """

        logdets = self.logdets
        self.logqw = - logdets
        """
        originally...
        logqw = - (0.5*(ep**2).sum(1)+0.5*T.log(2*np.pi)*num_params+logdets)
            --> constants are neglected in this wrapperfrom utils import log_laplace
        """
        self.logpw = self.prior(self.weights,0.,-T.log(self.lbda)).sum(1)
        """
        using normal prior centered at zero, with lbda being the inverse 
        of the variance
        """
        self.kl = (self.logqw - self.logpw).mean()
        if self.output_type == 'categorical':
            self.logpyx = - cc(self.y,self.target_var).mean()
        elif self.output_type == 'real':
            self.logpyx = - se(self.y,self.target_var).mean()
        else:
            assert False
        self.loss = - (self.logpyx - \
                       self.weight * self.kl/T.cast(self.dataset_size,floatX))

        # DK - extra monitoring
        params = self.params
        ds = self.dataset_size
        self.logpyx_grad = flatten_list(T.grad(-self.logpyx, params, disconnected_inputs='warn')).norm(2)
        self.logpw_grad = flatten_list(T.grad(-self.logpw.mean() / ds, params, disconnected_inputs='warn')).norm(2)
        self.logqw_grad = flatten_list(T.grad(self.logqw.mean() / ds, params, disconnected_inputs='warn')).norm(2)
        self.monitored = [self.logpyx, self.logpw, self.logqw,
                          self.logpyx_grad, self.logpw_grad, self.logqw_grad]
        
    def _get_grads(self):
        grads = T.grad(self.loss, self.params)
        mgrads = lasagne.updates.total_norm_constraint(grads,
                                                       max_norm=self.max_norm)
        cgrads = [T.clip(g, -self.clip_grad, self.clip_grad) for g in mgrads]
        if self.opt == 'adam':
            self.updates = lasagne.updates.adam(cgrads, self.params, 
                                                learning_rate=self.learning_rate)
        elif self.opt == 'momentum':
            self.updates = lasagne.updates.nesterov_momentum(cgrads, self.params, 
                                                learning_rate=self.learning_rate)
        elif self.opt == 'sgd':
            self.updates = lasagne.updates.sgd(cgrads, self.params, 
                                                learning_rate=self.learning_rate)
                                    
    def _get_train_func(self):
        inputs = [self.input_var,
                  self.target_var,
                  self.dataset_size,
                  self.learning_rate,
                  self.weight]
        train = theano.function(inputs,
                                self.loss,updates=self.updates)
        self.train_func_ = train
        # DK - putting this here, because is doesn't get overwritten by subclasses
        self.monitor_func = theano.function([self.input_var,
                                 self.target_var,
                                 self.dataset_size,
                                 self.learning_rate],
                                self.monitored,
                                on_unused_input='warn')
    
    def train_func(self,x,y,n,lr=lrdefault,w=1.0):
        return self.train_func_(x,y,n,lr,w)
        
    def _get_useful_funcs(self):
        pass
    
    
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

    def _init_pnet(self,init_batch):
        init_output = init_batch.copy()
        all_layers = lasagne.layers.get_all_layers(self.p_net)
        
        def stdize(layer,input):
            m = T.mean(input, layer.axes_to_sum)
            input -= m.dimshuffle(*layer.dimshuffle_args)
            stdv = T.sqrt(T.mean(T.square(input),axis=layer.axes_to_sum))
            input /= stdv.dimshuffle(*layer.dimshuffle_args)
            return -m/stdv, 1./stdv, input
            
        bs = list()
        gs = list()
        for l in all_layers[1:]:
            if isinstance(l,WeightNormLayer):
                b,g,init_output = stdize(l,init_output)
                bs.append(b)
                gs.append(g)
                if l.nonlinearity:
                    init_output = l.nonlinearity(init_output)
            else:
                init_output = l.get_output_for(init_output)
        
        new_gs = list()
        counter = 0
        for l in all_layers[1:]:
            if isinstance(l,WeightNormLayer):
                new_b = bs[counter].eval()
                new_g = gs[counter].eval()
                l.b.set_value(new_b)
                new_gs.append(new_g.reshape(-1))
                
                counter += 1
        
        gs_ = lasagne.layers.get_all_layers(self.h_net)[1].b
        new_gs = np.concatenate(new_gs)
        old_gs = gs_.get_value()
        gs_.set_value(new_gs*old_gs)

    
    

class MLPWeightNorm_BHN(Base_BHN):
    """
    Hypernet with dense coupling layer outputing posterior of rescaling 
    parameters of weightnorm MLP
    """

    
    def __init__(self,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 coupling=True,
                 n_hiddens=1,
                 n_units=200,
                 n_inputs=784,
                 n_classes=10,
                 output_type = 'categorical',
                 **kargs):
        
        self.__dict__.update(locals())

        self.weight_shapes = list()        
        self.weight_shapes.append((n_inputs,n_units))
        for i in range(1,n_hiddens):
            self.weight_shapes.append((n_units,n_units))
        self.weight_shapes.append((n_units,n_classes))
        self.num_params = sum(ws[1] for ws in self.weight_shapes)
        
        super(MLPWeightNorm_BHN, self).__init__(lbda=lbda,
                                                perdatapoint=perdatapoint,
                                                srng=srng,
                                                prior=prior,
                 output_type = output_type,
                                                **kargs)
    
    
    def _get_hyper_net(self):
        # inition random noise
        self.ep = self.srng.normal(size=(self.wd1,
                                    self.num_params),dtype=floatX)
        logdets_layers = []
        h_net = lasagne.layers.InputLayer([None,self.num_params])
        
        # mean and variation of the initial noise
        layer_temp = LinearFlowLayer(h_net)
        h_net = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        if self.flow == 'RealNVP':
            if self.coupling:
                layer_temp = CoupledDenseLayer(h_net,200)
                h_net = IndexLayer(layer_temp,0)
                logdets_layers.append(IndexLayer(layer_temp,1))
                for c in range(self.coupling-1):
                    h_net = PermuteLayer(h_net,self.num_params)
                    layer_temp = CoupledDenseLayer(h_net,200)
                    h_net = IndexLayer(layer_temp,0)
                    logdets_layers.append(IndexLayer(layer_temp,1))
        elif self.flow == 'IAF':
            layer_temp = IAFDenseLayer(h_net,200,1,L=self.coupling,cond_bias=False)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            assert False
        
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,self.ep)
        self.logdets = sum([get_output(ld,self.ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        t = 0 #np.cast['int32'](0) # TODO: what's wrong with np.cast
        p_net = lasagne.layers.InputLayer([None,self.n_inputs])
        inputs = {p_net:self.input_var}
        for ws in self.weight_shapes:
            # using weightnorm reparameterization
            # only need ws[1] parameters (for rescaling of the weight matrix)
            num_param = ws[1]
            weight = self.weights[:,t:t+num_param].reshape((self.wd1,ws[1]))
            p_net = lasagne.layers.DenseLayer(p_net,ws[1])
            p_net = stochastic_weight_norm(p_net,weight)
            print p_net.output_shape
            t += num_param
            
        if self.output_type == 'categorical':
            p_net.nonlinearity = nonlinearities.softmax
            y = T.clip(get_output(p_net,inputs), 0.001, 0.999) # stability
            self.p_net = p_net
            self.y = y
        elif self.output_type == 'real':
            p_net.nonlinearity = nonlinearities.linear
            y = get_output(p_net,inputs)
            self.p_net = p_net
            self.y = y
        else:
            assert False
        
    def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y.argmax(1))
        self.predict_fixed_mask = theano.function([self.input_var, self.weights],self.y)
        self.sample_weights = theano.function([], self.weights)
    
    def sample_qyx(self):
        """ return a function that will make predictions with a fixed random mask"""
        return lambda x : self.predict_fixed_mask(x, self.sample_weights())


class HyperWN_CNN(Base_BHN):
    """
    CHANGES:
        hypercnn for both mnist and cifar10

    """

    
    def __init__(self,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 coupling=4,
                 input_channels=3,
                 input_shape = (3,32,32),
                 n_classes=5,
                 n_convlayers=2,
                 n_channels=128,
                 kernel_size=3,
                 n_mlplayers=1,
                 n_units=1000,
                 stride=1,
                 pad='valid',
                 nonl=rectify,
                 pool_per=1,
                 **kargs):
        
        
        weight_shapes = list()
        args = list()
        n_channels = n_channels if isinstance(n_channels,list) else \
                     [n_channels for i in range(n_convlayers)]
        in_chan = input_channels
        for i in range(n_convlayers):
            out_chan = n_channels[i]
            weight_shape = (out_chan, in_chan, kernel_size, kernel_size)
            weight_shapes.append(weight_shape)
            
            num_filters = out_chan
            filter_size = kernel_size
            stride = stride
            pad = pad
            nonl = nonl
            # pool every `pool` conv layers
            if (i+1)%pool_per == 0:
                pool = 'max'
            else:
                pool = None
            arg = (num_filters,filter_size,stride,pad,nonl,pool)
            args.append(arg)
            in_chan = out_chan
        
        self.input_shape = input_shape
        self.weight_shapes = weight_shapes
        self.args = args
        self.num_classes = n_classes
        self.num_mlp_layers = n_mlplayers
        self.num_hids = n_units
        
        self.n_kernels = np.array(self.weight_shapes)[:,1].sum()
        self.kernel_shape = self.weight_shapes[0][:1]+self.weight_shapes[0][2:]
        print "kernel_shape", self.kernel_shape
        self.kernel_size = np.prod(self.weight_shapes[0])
    
        
        self.num_mlp_params = self.num_classes + \
                              self.num_hids * self.num_mlp_layers
        self.num_cnn_params = np.sum(np.array(self.weight_shapes)[:,0])
        self.num_params = self.num_mlp_params + self.num_cnn_params
        
        self.coupling = coupling
        super(HyperWN_CNN, self).__init__(lbda=lbda,
                                          perdatapoint=perdatapoint,
                                          srng=srng,
                                          prior=prior,
                                          **kargs)
    
    def _get_theano_variables(self):
        # redefine a 4-d tensor for convnet
        super(HyperWN_CNN, self)._get_theano_variables()
        self.input_var = T.tensor4('input_var')
     
    
    def _get_hyper_net(self):
        # inition random noise
        print self.num_params
        ep = self.srng.normal(size=(self.wd1,
                                    self.num_params),dtype=floatX)
        logdets_layers = []
        h_net = lasagne.layers.InputLayer([None,self.num_params])
        
        # mean and variation of the initial noise
        layer_temp = LinearFlowLayer(h_net)
        h_net = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        if self.flow == 'RealNVP':
            if self.coupling:
                layer_temp = CoupledWNDenseLayer(h_net,200)
                h_net = IndexLayer(layer_temp,0)
                logdets_layers.append(IndexLayer(layer_temp,1))
                 
                for c in range(self.coupling-1):
                    h_net = PermuteLayer(h_net,self.num_params)
                    
                    layer_temp = CoupledWNDenseLayer(h_net,200)
                    h_net = IndexLayer(layer_temp,0)
                    logdets_layers.append(IndexLayer(layer_temp,1))
        elif self.flow == 'IAF':
            layer_temp = IAFDenseLayer(h_net,200,1,
                                       L=self.coupling,cond_bias=False)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            assert False
        
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
        
        
    
    def _get_primary_net(self):
        
        t = 0 #np.cast['int32'](0)
        p_net = lasagne.layers.InputLayer((None,)+self.input_shape)
        print p_net.output_shape
        inputs = {p_net:self.input_var}
        for ws, args in zip(self.weight_shapes,self.args):

            num_filters = ws[0]
            
            # TO-DO: generalize to have multiple samples?
            weight = self.weights[0,t:t+num_filters].dimshuffle(0,'x','x','x')

            num_filters = args[0]
            filter_size = args[1]
            stride = args[2]
            pad = args[3]
            nonl = args[4]
            p_net = lasagne.layers.Conv2DLayer(p_net,num_filters,
                                               filter_size,stride,pad,
                                               nonlinearity=nonl)
            p_net = stochastic_weight_norm(p_net,weight)
            
            if args[5] == 'max':
                p_net = lasagne.layers.MaxPool2DLayer(p_net,2)
            #print p_net.output_shape
            t += num_filters

            
        for layer in range(self.num_mlp_layers):
            weight = self.weights[:,t:t+self.num_hids].reshape((self.wd1,
                                                                self.num_hids))
            p_net = lasagne.layers.DenseLayer(p_net,self.num_hids,
                                              nonlinearity=rectify)
            p_net = stochastic_weight_norm(p_net,weight)
            t += self.num_hids


        weight = self.weights[:,t:t+self.num_classes].reshape((self.wd1,self.num_classes))

        p_net = lasagne.layers.DenseLayer(p_net,self.num_classes,
                                          nonlinearity=nonlinearities.softmax)
        p_net = stochastic_weight_norm(p_net,weight)

        y = T.clip(get_output(p_net,inputs), 0.001, 0.999) # stability
        
        self.p_net = p_net
        self.y = y
        
    def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y.argmax(1))       
        


if __name__ == '__main__':
    
    
    
    # lenet 5
    model = HyperWN_CNN(lbda=1,
                        perdatapoint=False,
                        srng = RandomStreams(seed=427),
                        prior = log_normal,
                        coupling=4,
                        input_channels=3,
                        input_shape = (3,32,32),
                        n_classes=5,
                        n_convlayers=2,
                        n_channels=192,
                        kernel_size=3,
                        n_mlplayers=1,
                        n_units=1000,
                        stride=1,
                        pad='valid',
                        nonl=rectify,
                        pool_per=1)
    
    
    x = np.random.rand(8,3,32,32).astype('float32')
    y = np.zeros((8,5)).astype('float32')
    print model.train_func(x,y,1000)

    for l in lasagne.layers.get_all_layers(model.p_net)[1:]:
        if isinstance(l,WeightNormLayer):
            continue
        print l.output_shape
    
    
    
    
    
