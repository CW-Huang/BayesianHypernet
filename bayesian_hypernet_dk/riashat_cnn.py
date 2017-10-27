#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:49:51 2017

@author: Chin-Wei
"""

#from utils import log_normal, log_laplace
import numpy
import numpy as np

# utils issues...
def log_normal(x,mean,log_var,eps=0.0):
    c = - 0.5 * T.log(2*np.pi)
    return c - log_var/2. - (x - mean)**2 / (2. * T.exp(log_var) + eps)

import lasagne
import theano
import theano.tensor as T
floatX = theano.config.floatX

# DK
from bayesian_hypernet_dk.BHNs import MLPWeightNorm_BHN
from bayesian_hypernet_dk.BHNs import HyperCNN

from bayesian_hypernet_dk.helpers import SaveLoadMIXIN
from bayesian_hypernet_dk.helpers import load_mnist, load_cifar10






class RiashatCNN(SaveLoadMIXIN):
                     
    def __init__(self, dropout=None, opt='adam', **kwargs):

        self.__dict__.update(locals())
        ##################
        
        layer = lasagne.layers.InputLayer([None,1,28,28])
        layer = lasagne.layers.Conv2DLayer(layer, 32, 3, 1, 'valid', lasagne.nonlinearities.rectify)
        layer = lasagne.layers.Conv2DLayer(layer, 32, 3, 1, 'valid', lasagne.nonlinearities.rectify)
        layer = lasagne.layers.Pool2DLayer(layer, pool_size=5)
        if dropout:
            layer = lasagne.layers.dropout(layer, .25)
        # MLP layers
        layer = lasagne.layers.DenseLayer(layer, 256)
        l2_penalty = lasagne.regularization.regularize_layer_params_weighted({layer:3.5 / 128},
                lasagne.regularization.l2)
        if dropout:
            layer = lasagne.layers.dropout(layer, .5)
        layer = lasagne.layers.DenseLayer(layer, 10)
        layer.nonlinearity = lasagne.nonlinearities.softmax

        self.input_var = T.tensor4('input_var')
        self.target_var = T.matrix('target_var')
        self.learning_rate = T.scalar('leanring_rate')
        self.dataset_size = T.scalar('dataset_size') # useless
        
        self.layer = layer
        self.y = lasagne.layers.get_output(layer,self.input_var)
        self.y_det = lasagne.layers.get_output(layer,self.input_var,
                                               deterministic=True)
        
        losses = lasagne.objectives.categorical_crossentropy(self.y,
                                                             self.target_var)
        self.loss = losses.mean() + self.dataset_size * 0.
        self.loss += l2_penalty
        self.params = lasagne.layers.get_all_params(self.layer)
        # reset! DEPRECATED... use add_reset, call_reset instead...
        params0 = lasagne.layers.get_all_param_values(self.layer)
        updates = {p:p0 for p, p0 in zip(self.params,params0)}
        self.reset = theano.function([],None, updates=updates)
        self.add_reset('init')

        if opt == 'adam':
            self.updates = lasagne.updates.adam(self.loss,self.params,
                                                self.learning_rate)
        elif opt == 'momentum':
            self.updates = lasagne.updates.nesterov_momentum(self.loss,self.params,
                                                self.learning_rate)
        elif opt == 'sgd':
            self.updates = lasagne.updates.sgd(self.loss,self.params,
                                                self.learning_rate)

        print '\tgetting train_func'
        self.train_func = theano.function([self.input_var,
                                           self.target_var,
                                           self.dataset_size,
                                           self.learning_rate],
                                          self.loss,
                                          updates=self.updates)
        
        print '\tgetting useful_funcs'
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y_det.argmax(1))
        self.predict_expected = theano.function([self.input_var],self.y_det)
        




        
