#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:46:38 2017

@author: Chin-Wei
"""

from modules import LinearFlowLayer, IndexLayer, PermuteLayer, ReverseLayer
from modules import CoupledDenseLayer, stochasticDenseLayer2
from utils import log_normal, log_stdnormal
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


from helpers import flatten_list


if 1:#def main():
    """
    MNIST example
    weight norm reparameterized MLP with prior on rescaling parameters
    """
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',type=str, default='CW', choices=['CW', 'Dan'])
    parser.add_argument('--bs',default=128,type=int)  
    parser.add_argument('--coupling', type=int, default=4)  
    parser.add_argument('--epochs', type=int, default=30)  
    parser.add_argument('--lrdecay',action='store_true')  
    parser.add_argument('--lr0',default=0.003,type=float)  
    parser.add_argument('--lbda',default=0.5,type=float)  
    parser.add_argument('--model', default='hnet', type=str, choices=['mlp', 'hnet', 'hnet2', 'dropout', 'dropout2', 'weight_uncertainty'])
    parser.add_argument('--perdatapoint',action='store_true')    
    parser.add_argument('--size',default=10000,type=int)  
    args = parser.parse_args()
    print args
    locals().update(args.__dict__)
    
    perdatapoint = args.perdatapoint
    lr0 = args.lr0
    lrdecay = args.lrdecay
    lbda = np.cast[floatX](args.lbda)
    bs = args.bs
    size = max(10,min(50000,args.size))
    clip_grad = 100
    max_norm = 100
    
    # load dataset
    filename = '/data/lisa/data/mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
    
    
    input_var = T.matrix('input_var')
    target_var = T.matrix('target_var')
    dataset_size = T.scalar('dataset_size')
    lr = T.scalar('lr') 
    
    # 784 -> 20 -> 10
    if arch == 'CW':
        weight_shapes = [(784, 200), (200,  10)]
    elif arch == 'Dan':
        if model in ['hnet2', 'dropout2']:
            weight_shapes = [(784, 512), (512, 512), (512,512), (512,  10)]
        else:
            weight_shapes = [(784, 256), (256, 256), (256,256), (256,  10)]
    
    if model == 'weight_uncertainty': 
        num_params = sum(np.prod(ws) for ws in weight_shapes)
    else:
        num_params = sum(ws[1] for ws in weight_shapes)

    if perdatapoint:
        wd1 = input_var.shape[0]
    else:
        wd1 = 1

    if model in ['hnet', 'hnet2', 'weight_uncertainty']:
        # stochastic hypernet    
        ep = srng.normal(std=0.01,size=(wd1,num_params),dtype=floatX)
        logdets_layers = []
        h_layer = lasagne.layers.InputLayer([None,num_params])
        
        layer_temp = LinearFlowLayer(h_layer)
        h_layer = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))

        for c in range(coupling):
            h_layer = ReverseLayer(h_layer,num_params)
            layer_temp = CoupledDenseLayer(h_layer,10)
            h_layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        
        weights = lasagne.layers.get_output(h_layer,ep)
        
        # primary net
        t = np.cast['int32'](0)
        layer = lasagne.layers.InputLayer([None,784])
        inputs = {layer:input_var}
        for ws in weight_shapes:
            if model == 'weight_uncertainty':
                num_param = np.prod(ws)
            else:
                num_param = ws[1]
            w_layer = lasagne.layers.InputLayer((None,num_param))
            # TODO: why is reshape needed??
            weight = weights[:,t:t+num_param].reshape((wd1,num_param))
            inputs[w_layer] = weight
            layer = stochasticDenseLayer2([layer,w_layer],num_param)
            print layer.output_shape
            t += num_param

            
        layer.nonlinearity = nonlinearities.softmax
        y = get_output(layer,inputs)
        y = T.clip(y, 0.001, 0.999) # stability 
        
        # loss terms
        logdets = sum([get_output(logdet,ep) for logdet in logdets_layers])
        logqw = - (0.5*(ep**2).sum(1) + 0.5*T.log(2*np.pi)*num_params + logdets)
        #logpw = log_normal(weights,0.,-T.log(lbda)).sum(1)
        logpw = log_stdnormal(weights).sum(1)
        kl = (logqw - logpw).mean()
        logpyx = - cc(y,target_var).mean()
        loss = - (logpyx - kl/T.cast(dataset_size,floatX))
        params = lasagne.layers.get_all_params([h_layer,layer])

    else:
        # filler
        h_layer = lasagne.layers.InputLayer([None, 784])
        # JUST primary net
        layer = lasagne.layers.InputLayer([None,784])
        inputs = {layer:input_var}
        for nn, ws in enumerate(weight_shapes):
            layer = lasagne.layers.DenseLayer(layer, ws[1])
            if nn < len(weight_shapes)-1 and model in ['dropout', 'dropout2']:
                layer = lasagne.layers.dropout(layer, .5)
            print layer.output_shape
        layer.nonlinearity = nonlinearities.softmax
        y = get_output(layer,inputs)
        #y = T.clip(y, 0.001, 0.999) # stability 
        loss = cc(y,target_var).mean()
        params = lasagne.layers.get_all_params([h_layer,layer])
        loss = loss + lasagne.regularization.l2(flatten_list(params)) * np.float32(1.e-5)

    
    # TRAIN FUNCTION
    grads = T.grad(loss, params)
    mgrads = lasagne.updates.total_norm_constraint(grads,
                                                   max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
    updates = lasagne.updates.adam(cgrads, params, 
                                   learning_rate=lr)
                                        
    train = theano.function([input_var,target_var,dataset_size,lr],
                            loss,updates=updates,
                            on_unused_input='warn')
    predict = theano.function([input_var],y.argmax(1))

    ##################
    # TRAIN
    X, Y = train_x[:size],train_y[:size]
    Xt, Yt = valid_x,valid_y
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
    #epochs = 30
    records=list()
    
    t = 0
    import time
    t0 = time.time()
    for e in range(epochs):
        
        if lrdecay:
            lr = lr0 * 10**(-e/float(epochs-1))
        else:
            lr = lr0         
            
        for i in range(N/bs):
            x = X[i*bs:(i+1)*bs]
            y = Y[i*bs:(i+1)*bs]
            
            loss = train(x,y,N,lr)
            
            #if t%100==0:
            t+=1
        print 'epoch: {} {}, loss:{}'.format(e,t,loss)
        tr_acc = (predict(X)==Y.argmax(1)).mean()
        te_acc = (predict(Xt)==Yt.argmax(1)).mean()
        print '\ttrain acc: {}'.format(tr_acc)
        print '\ttest acc: {}'.format(te_acc)
        print 'time', time.time() - t0
            
        records.append(loss)
        

    

