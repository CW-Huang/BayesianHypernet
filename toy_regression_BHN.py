# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:36:33 2017


@author: Chin-Wei

Toy regression example
"""




# TODO (DK): take the plotting and the data, put it in my exp script...


import numpy as np
import matplotlib.pyplot as plt
from BHNs import Base_BHN
from modules import LinearFlowLayer, IndexLayer, PermuteLayer, ReverseLayer
from modules import CoupledDenseLayer, stochasticDenseLayer2
from utils import log_normal, log_laplace
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



class ToyRegression(Base_BHN):


    weight_shapes = [(1, 10),
                     (10,10),
                     (10, 2)]
    
    num_params = sum(ws[1] for ws in weight_shapes)
    
    def __init__(self,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 prior = log_normal,
                 coupling = True,
                 mixing = 'permute'):
        
        self.coupling = coupling
        self.__dict__.update(locals())
        super(ToyRegression, self).__init__(lbda=lbda,
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
        layer_temp = LinearFlowLayer(h_net,
                                     W=lasagne.init.Normal(0.01,-7))
        h_net = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        if self.coupling:
            # add more to introduce more correlation if needed
            layer_temp = CoupledDenseLayer(h_net,10)
            h_net = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
            
            for c in range(self.coupling-1):
                if self.mixing == 'permute':
                    h_net = PermuteLayer(h_net,self.num_params)
                elif self.mixing == 'reverse':
                    h_net = ReverseLayer(h_net,self.num_params)
                
                layer_temp = CoupledDenseLayer(h_net,10)
                h_net = IndexLayer(layer_temp,0)
                logdets_layers.append(IndexLayer(layer_temp,1))
                        
        # FINAL scale and shift (TODO: optional)
        layer_temp = LinearFlowLayer(h_net,
                                     W=lasagne.init.Normal(1.,0))
        h_net = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        
        self.h_net = h_net
        self.weights = lasagne.layers.get_output(h_net,ep)
        self.logdets = sum([get_output(ld,ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        t = np.cast['int32'](0)
        p_net = lasagne.layers.InputLayer([None,1])
        inputs = {p_net:self.input_var}
        for ws in self.weight_shapes:
            # using weightnorm reparameterization
            # only need ws[1] parameters (for rescaling of the weight matrix)
            num_param = ws[1]
            w_layer = lasagne.layers.InputLayer((None,ws[1]))
            weight = self.weights[:,t:t+num_param].reshape((self.wd1,ws[1]))
            inputs[w_layer] = weight
            p_net = stochasticDenseLayer2([p_net,w_layer],ws[1],
                                          nonlinearity=nonlinearities.tanh)
            print p_net.output_shape
            t += num_param
            
        p_net.nonlinearity = nonlinearities.linear  # replace the nonlinearity
                                                    # of the last layer
                                                    # with softmax for
                                                    # classification
        
        y = get_output(p_net,inputs)
        
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
        y_, lv = self.y[:,:1], self.y[:,1:]
        
        logpyx = log_normal(y_,self.target_var,lv).mean()
        self.loss = - (logpyx - kl/T.cast(self.dataset_size,floatX))
        
    def _get_useful_funcs(self):
        self.predict = theano.function([self.input_var],self.y[:,:1])
        sp = T.matrix('sp')
        predict_sp = self.y[:,:1] + sp * T.exp(0.5*self.y[:,1:])
        self.predict_sp = theano.function([self.input_var,sp],predict_sp)

    def _get_grads(self):
        grads = T.grad(self.loss, self.params)
        mgrads = lasagne.updates.total_norm_constraint(grads,
                                                       max_norm=self.max_norm)
        cgrads = [T.clip(g, -self.clip_grad, self.clip_grad) for g in mgrads]
        self.updates = lasagne.updates.adam(cgrads, self.params, 
                                           learning_rate=self.learning_rate)




                                            
# toy dataset
n = 5000
left1, right1 = 6,8
left2, right2 = 9,12
x1 = np.random.uniform(left1,right1,(n/2,1)).astype(floatX)
x2 = np.random.uniform(left2,right2,(n/2,1)).astype(floatX)
x = np.concatenate([x1,x2],0)
ep1 = np.random.randn(n,1).astype(floatX)
f = lambda x:0.01 * np.sin(2.5*x) * x**1.5 + 0.1 * x + ep1/(0.5*x)**2 - 1
t = f(x)
# 
n_ = 1000
f_ = lambda x:0.01 * np.sin(2.5*x) * x**1.5 + 0.1*x - 1
xx = np.linspace(1,20,n_).astype(floatX).reshape(n_,1)
yy = f_(xx)


model = ToyRegression(2.,coupling=4,prior=log_normal, mixing='reverse')


###############
# TRAIN
lr0 = 0.005
epochs=10000

plt.ion()

for i in range(epochs):
    lr = lr0 * 10**(-i/float(epochs-1))
    l = model.train_func(x,t,n,lr)
    if i%250==0:
        print i,l
        if 1: # interactive plotting
            # SAMPLES
            n_mc = 1000
            mc = np.zeros((n_,n_mc))
            for i in range(n_mc):
                sp = np.random.randn(n_,1).astype(floatX) 
                mc[:,i:i+1] = model.predict_sp(xx,sp)
            # PLOTTING
            plt.clf()
            plot2 = plt.fill_between(xx[:,0], 
                                     mc.mean(1)-mc.std(1,ddof=1), 
                                     mc.mean(1)+mc.std(1,ddof=1),
                                     facecolor='gray')
                               
                               
            plot2 = plt.plot(xx, mc.mean(1), 'r-')                 
            plot = plt.scatter(x,t)
            plot1 = plt.plot(xx,yy,'y--')


            plt.vlines(left1,yy.min(),yy.max())
            plt.vlines(right1,yy.min(),yy.max())

            plt.vlines(left2,yy.min(),yy.max())
            plt.vlines(right2,yy.min(),yy.max())
            plt.pause(.01)

y_ = model.predict(x)




###############
# SAMPLES
n_mc = 1000
mc = np.zeros((n_,n_mc))
for i in range(n_mc):
    sp = np.random.randn(n_,1).astype(floatX) 
    mc[:,i:i+1] = model.predict_sp(xx,sp)




###############
# PLOTTING
plt.figure()

plot2 = plt.fill_between(xx[:,0], 
                         mc.mean(1)-mc.std(1,ddof=1), 
                         mc.mean(1)+mc.std(1,ddof=1),
                         facecolor='gray')
                   
                   
plot2 = plt.plot(xx, mc.mean(1), 'r-')                 
plot = plt.scatter(x,t)
plot1 = plt.plot(xx,yy,'y--')


plt.vlines(left1,yy.min(),yy.max())
plt.vlines(right1,yy.min(),yy.max())

plt.vlines(left2,yy.min(),yy.max())
plt.vlines(right2,yy.min(),yy.max())


