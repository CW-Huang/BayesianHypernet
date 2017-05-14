# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:46:38 2017

@author: Chin-Wei
"""

from modules import LinearFlowLayer, IndexLayer, PermuteLayer
from modules import CoupledDenseLayer, stochasticConv2DLayer, stochasticDenseLayer2
from utils import log_normal, log_stdnormal
from ops import load_mnist
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=427)
floatX = theano.config.floatX

import lasagne
from lasagne import nonlinearities
from lasagne.layers import get_output
from lasagne.objectives import categorical_crossentropy as cc
import numpy as np



def train_model(train_func,predict_func,X,Y,Xt,Yt,
                lr0=0.1,lrdecay=1,bs=20):
    
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
    epochs = 50
    records=list()
    
    t = 0
    for e in range(epochs):
        
        if lrdecay:
            lr = lr0 * 10**(-e/float(epochs-1))
        else:
            lr = lr0         
            
        for i in range(N/bs):
            x = X[i*bs:(i+1)*bs]
            y = Y[i*bs:(i+1)*bs]
            
            loss = train_func(x,y,N,lr)
            
            if t%100==0:
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)
                tr_acc = (predict_func(X)==Y.argmax(1)).mean()
                te_acc = (predict_func(Xt)==Yt.argmax(1)).mean()
                print '\ttrain acc: {}'.format(tr_acc)
                print '\ttest acc: {}'.format(te_acc)
            t+=1
            
        records.append(loss)
        
    return records



def main():
    """
    MNIST example
    weight norm reparameterized MLP with prior on rescaling parameters
    """
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--coupling',action='store_true')  
    parser.add_argument('--size',default=10000,type=int)  
    parser.add_argument('--lrdecay',action='store_true')  
    parser.add_argument('--lr0',default=0.1,type=float)  
    parser.add_argument('--lbda',default=0.01,type=float)  
    parser.add_argument('--bs',default=50,type=int)  
    args = parser.parse_args()
    print args
    
    coupling = args.coupling
    lr0 = args.lr0
    lrdecay = args.lrdecay
    lbda = np.cast[floatX](args.lbda)
    bs = args.bs
    size = max(10,min(50000,args.size))
    clip_grad = 5
    max_norm = 10
    
    # load dataset
    filename = '/data/lisa/data/mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
    train_x = train_x.reshape(50000,1,28,28)
    valid_x = valid_x.reshape(10000,1,28,28)
    test_x = test_x.reshape(10000,1,28,28)
        
    
    input_var = T.tensor4('input_var')
    target_var = T.matrix('target_var')
    dataset_size = T.scalar('dataset_size')
    lr = T.scalar('lr') 
    
    # 784 -> 20 -> 10
    weight_shapes = [(16,1,5,5),        # -> (None, 16, 14, 14)
                     (16,16,5,5),       # -> (None, 16,  7,  7)
                     (16,16,5,5)]       # -> (None, 16,  4,  4)

    
    num_params = sum(np.prod(ws) for ws in weight_shapes) + 10
    wd1 = 1

    # stochastic hypernet    
    ep = srng.normal(std=0.01,size=(wd1,num_params),dtype=floatX)
    logdets_layers = []
    h_layer = lasagne.layers.InputLayer([None,num_params])
    
    layer_temp = LinearFlowLayer(h_layer)
    h_layer = IndexLayer(layer_temp,0)
    logdets_layers.append(IndexLayer(layer_temp,1))
    
    if coupling:
        layer_temp = CoupledDenseLayer(h_layer,200)
        h_layer = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        h_layer = PermuteLayer(h_layer,num_params)
        
        layer_temp = CoupledDenseLayer(h_layer,200)
        h_layer = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
    
    weights = lasagne.layers.get_output(h_layer,ep)
    
    # primary net
    t = np.cast['int32'](0)
    layer = lasagne.layers.InputLayer([None,1,28,28])
    inputs = {layer:input_var}
    for ws in weight_shapes:
        num_param = np.prod(ws)
        weight = weights[:,t:t+num_param].reshape(ws)
        num_filters = ws[0]
        filter_size = ws[2]
        stride = 2
        pad = 'same'
        layer = stochasticConv2DLayer([layer,weight],
                                      num_filters, filter_size, stride, pad)
        print layer.output_shape
        t += num_param
    
    w_layer = lasagne.layers.InputLayer((None,10))
    weight = weights[:,t:t+10].reshape((wd1,10))
    inputs[w_layer] = weight
    layer = stochasticDenseLayer2([layer,w_layer],10,
                                  nonlinearity=nonlinearities.softmax)

    y = T.clip(get_output(layer,inputs), 0.001, 0.999)
    
    # loss terms
    logdets = sum([get_output(logdet,ep) for logdet in logdets_layers])
    logqw = - (0.5*(ep**2).sum(1) + 0.5*T.log(2*np.pi)*num_params + logdets)
    logpw = log_normal(weights,0.,-T.log(lbda)).sum(1)
    #logpw = log_stdnormal(weights).sum(1)
    kl = (logqw - logpw).mean()
    logpyx = - cc(y,target_var).mean()
    loss = - (logpyx - kl/T.cast(dataset_size,floatX))
    
    params = lasagne.layers.get_all_params([layer])[1:] # excluding rand state
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
                          valid_x,valid_y,
                          lr0,lrdecay,bs)
    
    
    
    output_probs = theano.function([input_var],y)
    MCt = np.zeros((100,1000,10))
    MCv = np.zeros((100,1000,10))
    for i in range(100):
        MCt[i] = output_probs(train_x[:1000])
        MCv[i] = output_probs(valid_x[:1000])
        
    tr = np.equal(MCt.mean(0).argmax(-1),train_y[:1000].argmax(-1)).mean()
    va = np.equal(MCv.mean(0).argmax(-1),valid_y[:1000].argmax(-1)).mean()
    print "train perf=", tr
    print "valid perf=", va


    for ii in range(15): 
        print np.round(MCt[ii][0] * 1000)
        
    
if __name__ == '__main__':
    main()

