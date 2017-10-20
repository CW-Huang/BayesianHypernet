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

# used in hyperCNN.py
class Conv2D_BHN(Base_BHN):
    """
    Hypernet with dense coupling layer outputing posterior of weight
    parameters of filters for Conv2D layer. 
    
    The last layer is fully connected with weightnorm reparameterization 
    as in `MLPWeightNorm_BHN`. 
    """

    weight_shapes = [(16,1,5,5),        # -> (None, 16, 14, 14)
                     (16,16,5,5),       # -> (None, 16,  7,  7)
                     (16,16,5,5)]       # -> (None, 16,  4,  4)
    
    # [num_filters, filter_size, stride, pad, nonlinearity]
    # needs to be consistent with weight_shapes
    args = [[16,5,2,'same',rectify],
            [16,5,2,'same',rectify],
            [16,5,2,'same',rectify]]

    num_classes = 10
    num_params = sum(np.prod(ws) for ws in weight_shapes) + num_classes
                                                            # 10 classes
                                                            # need to be
                                                            # specified in 
                                                            # _get_primary_net

    
    def __init__(self,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 coupling = 1,
                 **kargs):
        
        self.coupling = coupling
        super(Conv2D_BHN, self).__init__(lbda=lbda,
                                         perdatapoint=perdatapoint,
                                         srng=srng,
                                         prior=prior,
                                         **kargs)
    
    def _get_theano_variables(self):
        # redefine a 4-d tensor for convnet
        self.input_var = T.tensor4('input_var')
        self.target_var = T.matrix('target_var')
        self.dataset_size = T.scalar('dataset_size')
        self.learning_rate = T.scalar('learning_rate') 
     
    
    def _get_hyper_net(self):
        # inition random noise
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
                # add more to introduce more correlation if needed
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
            layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            assert False
        
        
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        nc = self.num_classes
        t = np.cast['int32'](0)
        p_net = lasagne.layers.InputLayer([None,1,28,28])
        inputs = {p_net:self.input_var}
        for ws, args in zip(self.weight_shapes,self.args):
            num_param = np.prod(ws)
            weight = self.weights[:,t:t+num_param].reshape(ws)
            num_filters = args[0]
            filter_size = args[1]
            stride = args[2]
            pad = args[3]
            nonl = args[4]
            p_net = stochasticConv2DLayer([p_net,weight],
                                          num_filters,filter_size,stride,pad,
                                          nonlinearity=nonl)
            print p_net.output_shape
            t += num_param
        
        w_layer = lasagne.layers.InputLayer((None,nc))
        weight = self.weights[:,t:t+nc].reshape((self.wd1,nc))
        inputs[w_layer] = weight
        p_net = stochasticDenseLayer2([p_net,w_layer],nc,
                                      nonlinearity=nonlinearities.softmax)

        y = T.clip(get_output(p_net,inputs), 0.001, 0.999) # stability
        
        self.p_net = p_net
        self.y = y
        
    def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y.argmax(1))
        

# TODO: unused
class Conv2D_shared_BHN(Base_BHN):
    """
    Hypernet with dense coupling layer outputing posterior of weight
    parameters of filters for Conv2D layer. 
    
    The last layer is fully connected with weightnorm reparameterization 
    as in `MLPWeightNorm_BHN`. 
    """

    weight_shapes = [(16,1,5,5),        # -> (None, 16, 14, 14)
                     (16,16,5,5),       # -> (None, 16,  7,  7)
                     (16,16,5,5)]       # -> (None, 16,  4,  4)
    
    n_kernels = np.array(weight_shapes)[:,1].sum()
    kernel_shape = weight_shapes[0][:1]+weight_shapes[0][2:]
                                        # make sure it's the same 
                                        # across kernels
    
    # [num_filters, filter_size, stride, pad, nonlinearity]
    # needs to be consistent with weight_shapes
    args = [[16,5,2,'same',rectify],
            [16,5,2,'same',rectify],
            [16,5,2,'same',rectify]]
    
    num_classes = 10
    num_params = sum(np.prod(ws) for ws in weight_shapes) + num_classes
                                                            # 10 classes
                                                            # need to be
                                                            # specified in 
                                                            # _get_primary_net

    
    def __init__(self,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 coupling = 1,
                 **kargs):
        
        self.coupling = coupling
        super(Conv2D_shared_BHN, self).__init__(lbda=lbda,
                                                perdatapoint=perdatapoint,
                                                srng=srng,
                                                prior=prior,
                                                **kargs)
    
    def _get_theano_variables(self):
        # redefine a 4-d tensor for convnet
        self.input_var = T.tensor4('input_var')
        self.target_var = T.matrix('target_var')
        self.dataset_size = T.scalar('dataset_size')
        self.learning_rate = T.scalar('learning_rate') 
     
    
    def _get_hyper_net(self):
        # inition random noise
        ep = self.srng.normal(size=(1,
                                    self.num_params),dtype=floatX)
        logdets_layers = []
        h_net = lasagne.layers.InputLayer([1,self.num_params])
        
        # mean and variation of the initial noise
        layer_temp = LinearFlowLayer(h_net)
        h_net = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        # split the noise: hnet1 for filters, hnet2 for WN params (DK)
        h_net = SplitLayer(h_net,self.num_params-self.num_classes,1)
        h_net1 = IndexLayer(h_net,0,(1,self.num_params-self.num_classes))
        # TODO: full h_net2
        h_net2 = IndexLayer(h_net,1)

        h_net1 = lasagne.layers.ReshapeLayer(h_net1,
                                             (self.n_kernels,) + \
                                             (np.prod(self.kernel_shape),))
        if self.flow == 'RealNVP':
            if self.coupling:
                layer_temp = CoupledDenseLayer(h_net1,100)
                h_net1 = IndexLayer(layer_temp,0)
                logdets_layers.append(IndexLayer(layer_temp,1))
                
                for c in range(self.coupling-1):
                    h_net1 = PermuteLayer(h_net1,self.num_params)
                    
                    layer_temp = CoupledDenseLayer(h_net1,100)
                    h_net1 = IndexLayer(layer_temp,0)
                    logdets_layers.append(IndexLayer(layer_temp,1))
        elif self.flow == 'IAF':
            layer_temp = IAFDenseLayer(h_net1,200,1,L=self.coupling,cond_bias=False)
            layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            assert False
        
        self.kernel_weights = lasagne.layers.get_output(h_net1,ep)
        h_net1 = lasagne.layers.ReshapeLayer(h_net1,
                                             (1, self.n_kernels * \
                                                 np.prod(self.kernel_shape) ) )
        h_net = lasagne.layers.ConcatLayer([h_net1,h_net2],1)
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        nc = self.num_classes
        t = np.cast['int32'](0)
        k = np.cast['int32'](0)
        p_net = lasagne.layers.InputLayer([None,1,28,28])
        inputs = {p_net:self.input_var}
        for ws, args in zip(self.weight_shapes,self.args):
            num_param = np.prod(ws)
            num_kernel = ws[1]
            weight = self.kernel_weights[k:k+num_kernel,:]
            weight = weight.dimshuffle(1,0).reshape(self.kernel_shape + \
                                                    (num_kernel,))
            weight = weight.dimshuffle(0,3,1,2)
            num_filters = args[0]
            filter_size = args[1]
            stride = args[2]
            pad = args[3]
            nonl = args[4]
            p_net = stochasticConv2DLayer([p_net,weight],
                                          num_filters,filter_size,stride,pad,
                                          nonlinearity=nonl)
            print p_net.output_shape
            t += num_param
            k += num_kernel
        
        w_layer = lasagne.layers.InputLayer((None,nc))
        weight = self.weights[:,t:t+nc].reshape((self.wd1,nc))
        inputs[w_layer] = weight
        p_net = stochasticDenseLayer2([p_net,w_layer],nc,
                                      nonlinearity=nonlinearities.softmax)

        y = T.clip(get_output(p_net,inputs), 0.001, 0.999) # stability
        
        self.p_net = p_net
        self.y = y
        
    def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y.argmax(1))
     
  


# TODO: unused
class Conv2D_BHN_AL(Base_BHN):
    """
    Hypernet with dense coupling layer outputing posterior of weight
    parameters of filters for Conv2D layer. 
    
    The last layer is fully connected with weightnorm reparameterization 
    as in `MLPWeightNorm_BHN`. 
    """

    weight_shapes = [(32,1,3,3),        # -> (None, 16, 14, 14)
                     (32,32,3,3),       # -> (None, 16,  7,  7)
                     (32,32,3,3)]       # -> (None, 16,  4,  4)
    
    # [num_filters, filter_size, stride, pad, nonlinearity]
    # needs to be consistent with weight_shapes
    args = [[32,3,2,'same',rectify],
            [32,3,2,'same',rectify],
            [32,3,2,'same',rectify]]

    num_classes = 10
    num_params = sum(np.prod(ws) for ws in weight_shapes) + num_classes
                                                            # 10 classes
                                                            # need to be
                                                            # specified in 
                                                            # _get_primary_net

    
    def __init__(self,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 coupling=True,
                 **kargs):
        
        self.coupling = coupling
        super(Conv2D_BHN_AL, self).__init__(lbda=lbda,
                                         perdatapoint=perdatapoint,
                                         srng=srng,
                                         prior=prior,
                                         **kargs)
    
    def _get_theano_variables(self):
        # redefine a 4-d tensor for convnet
        self.input_var = T.tensor4('input_var')
        self.target_var = T.matrix('target_var')
        self.dataset_size = T.scalar('dataset_size')
        self.learning_rate = T.scalar('learning_rate') 
     
    
    def _get_hyper_net(self):
        # inition random noise
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
            layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            assert False
        
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        nc = self.num_classes
        t = np.cast['int32'](0)
        p_net = lasagne.layers.InputLayer([None,1,28,28])
        inputs = {p_net:self.input_var}
        for ws, args in zip(self.weight_shapes,self.args):
            num_param = np.prod(ws)
            weight = self.weights[:,t:t+num_param].reshape(ws)
            num_filters = args[0]
            filter_size = args[1]
            stride = args[2]
            pad = args[3]
            nonl = args[4]
            p_net = stochasticConv2DLayer([p_net,weight],
                                          num_filters,filter_size,stride,pad,
                                          nonlinearity=nonl)
            print p_net.output_shape
            t += num_param
        
        w_layer = lasagne.layers.InputLayer((None,nc))
        weight = self.weights[:,t:t+nc].reshape((self.wd1,nc))
        inputs[w_layer] = weight
        p_net = stochasticDenseLayer2([p_net,w_layer],nc,
                                      nonlinearity=nonlinearities.softmax)

        y = T.clip(get_output(p_net,inputs), 0.001, 0.999) # stability
        
        self.p_net = p_net
        self.y = y
        
    def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y.argmax(1))       
        





################################################33
# CW heavily modified version of below
# TODO: unused
class HyperCNN_CW(Base_BHN):
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
                 dataset='mnist',
                 **kargs):
        
        self.dataset = dataset
        
        if dataset == 'mnist':
            self.weight_shapes = [(32,1,3,3),        # -> (None, 16, 14, 14)
                                  (32,32,3,3),       # -> (None, 16,  7,  7)
                                  (32,32,3,3)]       # -> (None, 16,  4,  4)
            self.args = [[32,3,1,'same',rectify,'max'],
                         [32,3,1,'same',rectify,'max'],
                         [32,3,1,'same',rectify,'max']]
            
            self.num_classes = 10
            self.num_hids = 128
            self.num_mlp_layers = 1

        elif dataset == 'cifar10':
            self.weight_shapes = [(64, 3,3,3),       
                                  (64,64,3,3),     
                                  (64,64,3,3),
                                  (64,64,3,3)]  
            self.args = [[64,3,1,'valid',rectify,None],
                         [64,3,1,'valid',rectify,'max'],
                         [64,3,1,'valid',rectify,None],
                         [64,3,1,'valid',rectify,'max']]
                                  
            self.num_classes = 10
            self.num_hids = 512
            self.num_mlp_layers = 1
            
            
        self.n_kernels = np.array(self.weight_shapes)[:,1].sum()
        self.kernel_shape = self.weight_shapes[0][:1]+self.weight_shapes[0][2:]
        print "kernel_shape", self.kernel_shape
        self.kernel_size = np.prod(self.weight_shapes[0])
    
        
        self.num_mlp_params = self.num_classes + \
                              self.num_hids * self.num_mlp_layers
        self.num_cnn_params = sum(np.prod(ws) for ws in self.weight_shapes)
        self.num_params = self.num_mlp_params + self.num_cnn_params
        
        self.coupling = coupling
        super(HyperCNN_CW, self).__init__(lbda=lbda,
                                          perdatapoint=perdatapoint,
                                          srng=srng,
                                          prior=prior,
                                          **kargs)
    
    def _get_theano_variables(self):
        # redefine a 4-d tensor for convnet
        self.input_var = T.tensor4('input_var')
        self.target_var = T.matrix('target_var')
        self.dataset_size = T.scalar('dataset_size')
        self.learning_rate = T.scalar('learning_rate') 
     
    
    def _get_hyper_net(self):
        # inition random noise
        ep = self.srng.normal(size=(self.wd1,
                                    self.num_params),dtype=floatX)
        logdets_layers = []
        h_net = lasagne.layers.InputLayer([None,self.num_params])
        
        # mean and variation of the initial noise
        layer_temp = LinearFlowLayer(h_net)
        h_net = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        # split the noise: hnet1 for filters, hnet2 for WN params (DK)
        h_net = SplitLayer(h_net,self.num_cnn_params,1)
        h_net1 = IndexLayer(h_net,0, (1, self.num_cnn_params))
        h_net2 = IndexLayer(h_net,1, (1, self.num_mlp_params))
        

        # CNN coupling
        h_net1 = lasagne.layers.ReshapeLayer(h_net1,
                                             (self.n_kernels,) + \
                                             (np.prod(self.kernel_shape),))

        if self.flow == 'RealNVP':
            if self.coupling:
                layer_temp = CoupledDenseLayer(h_net1,self.kernel_size )
                h_net1 = IndexLayer(layer_temp,0)
                logdets_layers.append(IndexLayer(layer_temp,1))

                for c in range(self.coupling-1):
                    h_net1 = ReverseLayer(h_net1,np.prod(self.kernel_shape))
                    
                    layer_temp = CoupledDenseLayer(h_net1,self.kernel_size )
                    h_net1 = IndexLayer(layer_temp,0)
                    logdets_layers.append(IndexLayer(layer_temp,1))
        elif self.flow == 'IAF':
            layer_temp = IAFDenseLayer(h_net1,200,1,L=self.coupling,cond_bias=False)
            layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            assert False

        h_net1 = lasagne.layers.ReshapeLayer(h_net1,
                                             (1, self.n_kernels *
                                                 np.prod(self.kernel_shape) ) )
                                                 
        layer_temp = ConvexBiasLayer(h_net1)
        h_net1 = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))

        
        h_net1_w = lasagne.layers.ReshapeLayer(h_net1,
                                               (self.n_kernels,
                                                np.prod(self.kernel_shape) ) )
             
        self.kernel_weights = lasagne.layers.get_output(h_net1_w,ep)
        

        # MLP coupling
        if self.flow == 'RealNVP':
            if self.coupling:
                layer_temp = CoupledDenseLayer(h_net2,self.num_mlp_params )
                h_net2 = IndexLayer(layer_temp,0)
                logdets_layers.append(IndexLayer(layer_temp,1))
                
                for c in range(self.coupling-1):
                    h_net2 = ReverseLayer(h_net2,self.num_mlp_params)
                    
                    layer_temp = CoupledDenseLayer(h_net2,self.num_mlp_params )
                    h_net2 = IndexLayer(layer_temp,0)
                    logdets_layers.append(IndexLayer(layer_temp,1))
        elif self.flow == 'IAF':
            layer_temp = IAFDenseLayer(h_net2,200,1,L=self.coupling,cond_bias=False)
            layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            assert False

        
        h_net = lasagne.layers.ConcatLayer([h_net1,h_net2],1)
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        nwn = self.num_mlp_params
        t = np.cast['int32'](0)
        k = np.cast['int32'](0)
        if self.dataset == 'mnist':
            p_net = lasagne.layers.InputLayer([None,1,28,28])
        elif self.dataset == 'cifar10':
            p_net = lasagne.layers.InputLayer([None,3,32,32])
        print p_net.output_shape
        inputs = {p_net:self.input_var}
        for ws, args in zip(self.weight_shapes,self.args):
            num_param = np.prod(ws)
            num_kernel = ws[1]
            weight = self.kernel_weights[k:k+num_kernel,:]
            weight = weight.dimshuffle(1,0).reshape(self.kernel_shape + \
                                                    (num_kernel,))

            weight = weight.dimshuffle(0,3,1,2)

            num_filters = args[0]
            filter_size = args[1]
            stride = args[2]
            pad = args[3]
            nonl = args[4]
            p_net = stochasticConv2DLayer([p_net,weight],
                                          num_filters,filter_size,stride,pad,
                                          nonlinearity=nonl)
            
            if args[5] == 'max':
                p_net = lasagne.layers.MaxPool2DLayer(p_net,2)
            print p_net.output_shape
            t += num_param
            k += num_kernel

            

        assert self.num_mlp_layers == 1
        for layer in range(self.num_mlp_layers):
            w_layer = lasagne.layers.InputLayer((None,self.num_hids))
            weight = self.weights[:,t:t+self.num_hids].reshape((self.wd1,self.num_hids))
            inputs[w_layer] = weight
            p_net = stochasticDenseLayer2([p_net,w_layer],self.num_hids,
                                          nonlinearity=nonlinearities.rectify)
            t += self.num_hids
            
        w_layer = lasagne.layers.InputLayer((None,self.num_classes))
        weight = self.weights[:,t:t+self.num_classes].reshape((self.wd1,self.num_classes))
        inputs[w_layer] = weight
        p_net = stochasticDenseLayer2([p_net,w_layer],self.num_classes,
                                      nonlinearity=nonlinearities.softmax)

        y = T.clip(get_output(p_net,inputs), 0.001, 0.999) # stability
        
        self.p_net = p_net
        self.y = y
        
    def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y.argmax(1))       
        
        
        




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
                 dataset='mnist',
                 **kargs):
        
        self.dataset = dataset
        
        if dataset == 'mnist':
            self.weight_shapes = [(32,1,3,3),        # -> (None, 16, 14, 14)
                                  (32,32,3,3),       # -> (None, 16,  7,  7)
                                  (32,32,3,3)]       # -> (None, 16,  4,  4)
            self.args = [[32,3,1,'same',rectify,'max'],
                         [32,3,1,'same',rectify,'max'],
                         [32,3,1,'same',rectify,'max']]
            
            self.num_classes = 10
            self.num_hids = 128
            self.num_mlp_layers = 1

        elif dataset == 'cifar10':
            self.weight_shapes = [(64, 3,3,3),       
                                  (64,64,3,3),     
                                  (128,64,3,3),
                                  (128,128,3,3)]
            self.args = [[64,3,1,'valid',rectify,None],
                         [64,3,1,'valid',rectify,'max'],
                         [128,3,1,'valid',rectify,None],
                         [128,3,1,'valid',rectify,'max']]
                                  
            self.num_classes = 10
            self.num_hids = 512
            self.num_mlp_layers = 1
            
            
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
        self.input_var = T.tensor4('input_var')
        self.target_var = T.matrix('target_var')
        self.dataset_size = T.scalar('dataset_size')
        self.learning_rate = T.scalar('learning_rate') 
     
    
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
            layer_temp = IAFDenseLayer(h_net,200,1,L=self.coupling,cond_bias=False)
            layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            assert False
        
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        
        t = np.cast['int32'](0)
        if self.dataset == 'mnist':
            p_net = lasagne.layers.InputLayer([None,1,28,28])
        elif self.dataset == 'cifar10':
            p_net = lasagne.layers.InputLayer([None,3,32,32])
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
        






################################################33
################################################33
################################################33
################################################33
################################################33
################################################33
################################################33
# DK heavily modified version of above
# CW: weightnorm-reparamed
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
class HyperCNN(Base_BHN):
    """
    CHANGES:
        share the hyperCNN
        add dense 128 layer
        reverse instead of permute
        change arguments to coupling layers to match kernel_size
        coupling layers for WN params
        default to 4 coupling layers
        provide shape to h_net2

    """

    
    def __init__(self,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 opt='adam',
                 coupling=4,
                 pad='same',
                 stride=2,
                 pool=None,
                 kernel_width=None,
                 dataset='mnist',
                 extra_linear=0,
                 **kargs):
        
        self.dataset = dataset
        
        if dataset == 'mnist':
            self.weight_shapes = [(32,1,3,3),        # -> (None, 16, 14, 14)
                                  (32,32,3,3),       # -> (None, 16,  7,  7)
                                  (32,32,3,3)]       # -> (None, 16,  4,  4)
        elif dataset == 'cifar10':
            self.weight_shapes = [(32,3,5,5),        # -> (None, 16, 16, 16)
                                  (32,32,5,5),       # -> (None, 16,  8,  8)
                                  (32,32,5,5)]       # -> (None, 16,  4,  4)
        # [num_filters, filter_size, stride, pad, nonlinearity]
        # needs to be consistent with weight_shapes
        if dataset == 'mnist':            
            self.args = [[32,3,stride,pad, rectify, pool],
                         [32,3,stride,pad, rectify, pool],
                         [32,3,stride,pad, rectify, pool]]
        elif dataset == 'cifar10':
            self.args = [[32,5,stride,pad, rectify, pool],
                         [32,5,stride,pad, rectify, pool],
                         [32,5,stride,pad, rectify, pool]]



        if kernel_width is not None: # OVERRIDE dataset argument!!!
            # ONLY FOR MNIST ACTIVE LEARNING EXPERIMENT!
            self.weight_shapes = [(32,1,kernel_width,kernel_width),        # -> (None, 16, 14, 14)
                                  (32,32,kernel_width,kernel_width),       # -> (None, 16,  7,  7)
                                  (32,32,kernel_width,kernel_width)]       # -> (None, 16,  4,  4)
            self.args = [[32,kernel_width,stride,pad, rectify, 'none'],
                         [32,kernel_width,stride,pad, rectify, 'none'],
                         [32,kernel_width,stride,pad, rectify, 'max']]
                                  

        self.n_kernels = np.array(self.weight_shapes)[:,1].sum()
        self.kernel_shape = self.weight_shapes[0][:1]+self.weight_shapes[0][2:]
        print "kernel_shape", self.kernel_shape
        self.kernel_size = np.prod(self.weight_shapes[0])
    
    
        self.num_classes = 10
        self.num_hids = 128
        self.num_mlp_layers = 1
        self.num_mlp_params = self.num_classes + \
                              self.num_hids * self.num_mlp_layers
        self.num_cnn_params = np.sum(np.array(self.weight_shapes)[:,0])
        self.num_params = self.num_mlp_params + self.num_cnn_params
        
        self.coupling = coupling
        self.extra_linear = extra_linear
        super(HyperCNN, self).__init__(lbda=lbda,
                                         perdatapoint=perdatapoint,
                                         srng=srng,
                                         opt=opt,
                                         prior=prior,
                                         **kargs)
    
    def _get_theano_variables(self):
        # redefine a 4-d tensor for convnet
        self.input_var = T.tensor4('input_var')
        self.target_var = T.matrix('target_var')
        self.dataset_size = T.scalar('dataset_size')
        self.learning_rate = T.scalar('learning_rate') 
     
    
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
            layer_temp = IAFDenseLayer(h_net,200,1,L=self.coupling,cond_bias=False)
            layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            assert False
        
        if self.extra_linear:
            layer_temp = ConvexBiasLayer(h_net)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
            
        
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        
        t = np.cast['int32'](0)
        if self.dataset == 'mnist':
            p_net = lasagne.layers.InputLayer([None,1,28,28])
        elif self.dataset == 'cifar10':
            p_net = lasagne.layers.InputLayer([None,3,32,32])
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
        

        


class BHN_Q_Network(Base_BHN):
    """
    Hypernet with dense coupling layer outputing posterior of rescaling 
    parameters of weightnorm MLP
    """
    # 784 -> 20 -> 10
    # weight_shapes = [(784, 200),
    #                  (200,  10)]
    

    weight_shapes = [(128, 32),
            (32, 2)         ]

    coupling_dim = 64

    num_params = sum(ws[1] for ws in weight_shapes)
    
    def __init__(self,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 wn=0,
                 weight_shapes=None,
                 coupling_dim=None,
                 coupling=True,
                 **kargs):
        
        self.wn = wn
        self.coupling = coupling
        super(BHN_Q_Network, self).__init__(lbda=lbda,
                                                perdatapoint=perdatapoint,
                                                srng=srng,
                                                prior=prior,
                                                **kargs)
        
        if weight_shapes is not None:
            self.weight_shapes = weight_shapes
        
        if coupling_dim is not None:
            self.coupling_dim = coupling_dim
    
    def _get_hyper_net(self):
        # inition random noise
        ep = self.srng.normal(size=(self.wd1,
                                    self.num_params),dtype=floatX)
        logdets_layers = []
        h_net = lasagne.layers.InputLayer([None,self.num_params])
        
        # mean and variation of the initial noise
        layer_temp = LinearFlowLayer(h_net)
        h_net = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        # This one (only) allows us to NOT use WN layer (but there's no real reason to do that FWICT)
        if self.flow == 'RealNVP':
            if self.coupling:
                if self.wn:
                    layer_temp = CoupledWNDenseLayer(h_net,self.coupling_dim)
                else:
                    layer_temp = CoupledDenseLayer(h_net,self.coupling_dim)
                h_net = IndexLayer(layer_temp,0)
                logdets_layers.append(IndexLayer(layer_temp,1))
                
                for c in range(self.coupling-1):
                    h_net = PermuteLayer(h_net,self.num_params)
                    
                    if self.wn:
                        layer_temp = CoupledWNDenseLayer(h_net,self.coupling_dim)
                    else:
                        layer_temp = CoupledDenseLayer(h_net,self.coupling_dim)
                    h_net = IndexLayer(layer_temp,0)
                    logdets_layers.append(IndexLayer(layer_temp,1))
        elif self.flow == 'IAF':
            layer_temp = IAFDenseLayer(h_net,200,1,L=self.coupling,cond_bias=False)
            layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        else:
            assert False
        
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        t = np.cast['int32'](0)
        #p_net = lasagne.layers.InputLayer([None,784])
        
        p_net = lasagne.layers.InputLayer([None,4])
        inputs = {p_net:self.input_var}
        for ws in self.weight_shapes:
            # using weightnorm reparameterization
            # only need ws[1] parameters (for rescaling of the weight matrix)
            num_param = ws[1]
            w_layer = lasagne.layers.InputLayer((None,ws[1]))
            weight = self.weights[:,t:t+num_param].reshape((self.wd1,ws[1]))
            inputs[w_layer] = weight
            p_net = stochasticDenseLayer2([p_net,w_layer],ws[1])
            
            #p_net = ConvexBiasLayer([p_net,w_layer],ws[1])
            print p_net.output_shape
            t += num_param
            


        p_net.nonlinearity = nonlinearities.linear 
        #p_net.nonlinearity = nonlinearities.softmax # replace the nonlinearity
                                                    # of the last layer
                                                    # with softmax for
                                                    # classification
        
        y = get_output(p_net,inputs) # stability
        
        self.p_net = p_net
        self.y = y
    
    def _get_elbo(self):
        """
        negative elbo, an upper bound on NLL
        """

        logdets = self.logdets
        logqw = - logdets
        """
        originally...
        logqw = - (0.5*(ep**2).sum(1)+0.5*T.log(2*np.pi)*num_params+logdets)
            --> constants are neglected in this wrapper
        """
        logpw = self.prior(self.weights,0.,-T.log(self.lbda)).sum(1)
        """
        using normal prior centered at zero, with lbda being the inverse 
        of the variance
        """
        kl = (logqw - logpw).mean()
        
        logpyx = - se(self.y,self.target_var).sum(1).mean()
        self.loss = - (logpyx - kl/T.cast(self.dataset_size,floatX))
        self.monitored = [self.loss]
        
        
    def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        #self.predict = theano.function([self.input_var],self.y.argmax(1))
        self.predict = theano.function([self.input_var],self.y)





if __name__ == '__main__':
    
    
    init_batch = np.random.rand(3,784).astype('float32') 
    model = MLPWeightNorm_BHN(lbda=1.,
                              perdatapoint=0,
                              prior=log_normal,
                              coupling=2,
                              n_hiddens=2,
                              n_units=32,
                              init_batch=init_batch)
    
    print model.predict_proba(init_batch)
    
    
    
    
    
    



