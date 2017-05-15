# -*- coding: utf-8 -*-
"""
Created on Sun May 14 17:58:58 2017

@author: Chin-Wei
"""


from modules import LinearFlowLayer, IndexLayer, PermuteLayer, SplitLayer
from modules import CoupledDenseLayer, stochasticDenseLayer2, \
                    stochasticConv2DLayer
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
import numpy as np




class Base_BHN(object):
    
    max_norm = 10
    clip_grad = 5
    
    def __init__(self,
                 lbda=1.,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal):
        
        self.lbda = lbda
        self.perdatapoint = perdatapoint
        self.srng = srng
        self.prior = prior
        
        
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

    
    def _get_theano_variables(self):
        self.input_var = T.matrix('input_var')
        self.target_var = T.matrix('target_var')
        self.dataset_size = T.scalar('dataset_size')
        self.learning_rate = T.scalar('learning_rate') 
        
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
        logqw = - logdets
        """
        originally...
        logqw = - (0.5*(ep**2).sum(1)+0.5*T.log(2*np.pi)*num_params+logdets)
            --> constants are neglected in this wrapperfrom utils import log_laplace
        """
        logpw = self.prior(self.weights,0.,-T.log(self.lbda)).sum(1)
        """
        using normal prior centered at zero, with lbda being the inverse 
        of the variance
        """
        kl = (logqw - logpw).mean()
        logpyx = - cc(self.y,self.target_var).mean()
        self.loss = - (logpyx - kl/T.cast(self.dataset_size,floatX))
    
    def _get_grads(self):
        grads = T.grad(self.loss, self.params)
        mgrads = lasagne.updates.total_norm_constraint(grads,
                                                       max_norm=self.max_norm)
        cgrads = [T.clip(g, -self.clip_grad, self.clip_grad) for g in mgrads]
        self.updates = lasagne.updates.adam(cgrads, self.params, 
                                            learning_rate=self.learning_rate)
                                    
    def _get_train_func(self):
        train = theano.function([self.input_var,
                                 self.target_var,
                                 self.dataset_size,
                                 self.learning_rate],
                                self.loss,updates=self.updates)
        self.train_func = train
        
    def _get_useful_funcs(self):
        pass


class MLPWeightNorm_BHN(Base_BHN):
    """
    Hypernet with dense coupling layer outputing posterior of rescaling 
    parameters of weightnorm MLP
    """
    # 784 -> 20 -> 10
    weight_shapes = [(784, 200),
                     (200,  10)]
    
    num_params = sum(ws[1] for ws in weight_shapes)
    
    def __init__(self,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 coupling=True):
        
        self.coupling = coupling
        super(MLPWeightNorm_BHN, self).__init__(lbda=lbda,
                                                perdatapoint=perdatapoint,
                                                srng=srng,
                                                prior=prior)
        
    
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
        
        if self.coupling:
            # add more to introduce more correlation if needed
            layer_temp = CoupledDenseLayer(h_net,200)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
            
            h_net = PermuteLayer(h_net,self.num_params)
            
            layer_temp = CoupledDenseLayer(h_net,200)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        t = np.cast['int32'](0)
        p_net = lasagne.layers.InputLayer([None,784])
        inputs = {p_net:self.input_var}
        for ws in self.weight_shapes:
            # using weightnorm reparameterization
            # only need ws[1] parameters (for rescaling of the weight matrix)
            num_param = ws[1]
            w_layer = lasagne.layers.InputLayer((None,ws[1]))
            weight = self.weights[:,t:t+num_param].reshape((self.wd1,ws[1]))
            inputs[w_layer] = weight
            p_net = stochasticDenseLayer2([p_net,w_layer],ws[1])
            print p_net.output_shape
            t += num_param
            
        p_net.nonlinearity = nonlinearities.softmax # replace the nonlinearity
                                                    # of the last layer
                                                    # with softmax for
                                                    # classification
        
        y = T.clip(get_output(p_net,inputs), 0.001, 0.999) # stability
        
        self.p_net = p_net
        self.y = y
        
    def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y.argmax(1))


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
                 coupling=True):
        
        self.coupling = coupling
        super(Conv2D_BHN, self).__init__(lbda=lbda,
                                         perdatapoint=perdatapoint,
                                         srng=srng,
                                         prior=prior)
    
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
        
        if self.coupling:
            # add more to introduce more correlation if needed
            layer_temp = CoupledDenseLayer(h_net,200)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
            
            h_net = PermuteLayer(h_net,self.num_params)
            
            layer_temp = CoupledDenseLayer(h_net,200)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        
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
                 coupling=True):
        
        self.coupling = coupling
        super(Conv2D_shared_BHN, self).__init__(lbda=lbda,
                                                perdatapoint=perdatapoint,
                                                srng=srng,
                                                prior=prior)
    
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
        
        h_net = SplitLayer(h_net,self.num_params-self.num_classes,1)
        h_net1 = IndexLayer(h_net,0,(1,self.num_params-self.num_classes))
        h_net2 = IndexLayer(h_net,1)


        h_net1 = lasagne.layers.ReshapeLayer(h_net1,
                                             (self.n_kernels,) + \
                                             (np.prod(self.kernel_shape),))
        if self.coupling:
            # add more to introduce more correlation if needed
            layer_temp = CoupledDenseLayer(h_net1,100)
            h_net1 = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
            
            h_net1 = PermuteLayer(h_net1,self.num_params)
            
            layer_temp = CoupledDenseLayer(h_net1,100)
            h_net1 = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        
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
        
        