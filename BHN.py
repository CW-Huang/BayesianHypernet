# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:46:38 2017

@author: Chin-Wei
"""

from modules import LinearFlowLayer, IndexLayer, PermuteLayer
from modules import CoupledDenseLayer, CoupledConv1DLayer
from utils import log_stdnormal
from ops import load_mnist
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=427)
floatX = theano.config.floatX

import lasagne
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import get_output
from lasagne.objectives import categorical_crossentropy as cc
import numpy as np



class stochasticDenseLayer(lasagne.layers.base.MergeLayer):
    
    def __init__(self, incomings, num_units, 
                 b=init.Constant(0.), nonlinearity=nonlinearities.tanh,
                 num_leading_axes=1, **kwargs):
        super(stochasticDenseLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.num_units = num_units
        
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)
    def get_output_shape_for(self,input_shapes):
        input_shape = input_shapes[0]
        weight_shape = input_shapes[1]
        return (input_shape[0], weight_shape[2])
        
    def get_output_for(self, inputs, **kwargs):
        """
        inputs[0].shape = (None, num_inputs)
        inputs[1].shape = (None/1, num_inputs, num_units)
        """
        input = inputs[0]
        W = inputs[1]
        activation = T.sum(input.dimshuffle(0,1,'x') * W, axis = 1)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)


def train_model(train_func,predict_func,X,Y,Xt,Yt,bs=20):
    
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
    epochs = 50
    records=list()
    
    t = 0
    for e in range(epochs):

        lr = 0.01 * 10**(-e/float(epochs-1))                    
        for i in range(N/bs):
            x = X[i*bs:(i+1)*bs]
            y = Y[i*bs:(i+1)*bs]
            
            loss = train_func(x,y,N,lr)
            
            if t%100==0:
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)
                acc = (predict_func(Xt)==Yt.argmax(1)).mean()
                print '\tacc: {}'.format(acc)
            t+=1
            
        records.append(loss)
        
    return records



def main():
    """
    MNIST example
    """
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--perdatapoint',default=0,type=bool)
    parser.add_argument('--coupling',default=0,type=bool)
    parser.add_argument('--size',default=10000,type=bool)
    args = parser.parse_args()
    print args

    perdatapoint = args.perdatapoint
    coupling = args.coupling
    size = max(10,min(50000,args.size))
    clip_grad = 100
    max_norm = 1000

    # load dataset
    filename = r'/data/lisa/data/mnist/mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)

    input_var = T.matrix('input_var')
    target_var = T.matrix('target_var')
    dataset_size = T.scalar('dataset_size')
    lr = T.scalar('lr') 
    
    # 784 -> 20 -> 10
    weight_shapes = [(784, 10),
                     (10,  10)]
    
    num_params = sum(np.prod(ws) for ws in weight_shapes)
    if perdatapoint:
        wd1 = input_var.shape[0]
    else:
        wd1 = 1

    # stochastic hypernet    
    ep = srng.normal(size=(wd1,num_params),dtype=floatX)
    logdets_layers = []
    h_layer = lasagne.layers.InputLayer([None,num_params])
    
    layer_temp = LinearFlowLayer(h_layer)
    h_layer = IndexLayer(layer_temp,0)
    logdets_layers.append(IndexLayer(layer_temp,1))
    
    if coupling:
        layer_temp = CoupledConv1DLayer(h_layer,16,5)
        h_layer = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        h_layer = PermuteLayer(h_layer,num_params)
        
        layer_temp = CoupledConv1DLayer(h_layer,16,5)
        h_layer = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
    
    weights = lasagne.layers.get_output(h_layer,ep)
    
    # primary net
    t = np.cast['int32'](0)
    layer = lasagne.layers.InputLayer([None,784])
    inputs = {layer:input_var}
    for ws in weight_shapes:
        num_param = np.prod(ws)
        print t, t+num_param
        w_layer = lasagne.layers.InputLayer((None,)+ws)
        weight = weights[:,t:t+num_param].reshape((wd1,)+ws)
        inputs[w_layer] = weight
        layer = stochasticDenseLayer([layer,w_layer],ws[1])
        t += num_param
        
    layer.nonlinearity = nonlinearities.softmax
    y = T.clip(get_output(layer,inputs), 0.001, 0.999) # stability 
    
    # loss terms
    logdets = sum([get_output(logdet,ep) for logdet in logdets_layers])
    logqw = - (0.5*(ep**2).sum(1) + 0.5*T.log(2*np.pi)*num_params + logdets)
    logpw = log_stdnormal(weights).sum(1)
    kl = (logqw - logpw).mean()
    logpyx = - cc(y,target_var).mean()
    loss = - (logpyx - kl/T.cast(dataset_size,floatX))
    
    params = lasagne.layers.get_all_params([h_layer,layer])
    grads = T.grad(loss, params)
    mgrads = lasagne.updates.total_norm_constraint(grads,
                                                   max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
    updates = lasagne.updates.adam(cgrads, params, 
                                   learning_rate=lr)
                                            
    
    train = theano.function([input_var,target_var,dataset_size,lr],
                            loss,updates=updates)
    predict = theano.function([input_var],y.argmax(1))
    
    records = train_model(train,predict,
                          train_x[:size],train_y[:size],
                          valid_x,valid_y)
    
    
    
if __name__ == '__main__':
    main()

    #TODO: 
    #g=theano.function([input_var,target_var],T.grad(-logyx,params[0]))
    #g(train_x[:2],train_y[:2])[:7840]  ----> all zeros if use > 2 layers!

