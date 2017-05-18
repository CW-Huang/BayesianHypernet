#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:46:38 2017

@author: Chin-Wei
"""

from modules import LinearFlowLayer, IndexLayer, PermuteLayer, ReverseLayer
from modules import CoupledDenseLayer, stochasticDenseLayer2, ConvexBiasLayer
from modules import * # just in case
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


from helpers import flatten_list, gelu, plot_dict


# TODO: 
#   AD in script
#   init

NUM_CLASSES = 10


if 1:#def main():
    """
    MNIST example
    weight norm reparameterized MLP with prior on rescaling parameters
    """
    
    import argparse
    import sys
    import os
    import numpy 
    np = numpy
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',type=str, default='Dan', choices=['CW', 'Dan'])
    parser.add_argument('--bs',default=128,type=int)  
    parser.add_argument('--convex_combination', type=int, default=0) 
    parser.add_argument('--coupling', type=int, default=4) 
    parser.add_argument('--epochs', type=int, default=100)  
    parser.add_argument('--init', type=str, default='normal')  
    parser.add_argument('--lrdecay',action='store_true')  
    parser.add_argument('--lr0',default=0.001,type=float)  
    parser.add_argument('--lbda',default=1.,type=float)  
    parser.add_argument('--model', default='mlp', type=str, choices=['mlp', 'hnet', 'hnet2', 'dropout', 'dropout2', 'weight_uncertainty'])
    parser.add_argument('--nonlinearity',default='rectify', type=str)
    parser.add_argument('--perdatapoint',action='store_true')    
    parser.add_argument('--num_examples',default=10000,type=int)  
    parser.add_argument('--num_samples',default=100,type=int)  
    parser.add_argument('--size',default=50000,type=int)  
    #
    #parser.add_argument('--save_path',default=None,type=str)  
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--verbose', type=int, default=1)


    # --------------------------------------------
    # PARSE ARGS and SET-UP SAVING and RANDOM SEED

    args = parser.parse_args()
    args_dict = args.__dict__

    # save_path = filename + PROVIDED parser arguments
    flags = [flag.lstrip('--') for flag in sys.argv[1:]]
    flags = [ff for ff in flags if not ff.startswith('save_dir')]
    save_dir = args_dict.pop('save_dir')
    save_path = os.path.join(save_dir, os.path.basename(__file__) + '___' + '_'.join(flags))
    args_dict['save_path'] = save_path

    if args_dict['save']:
        # make directory for results, save ALL parser arguments
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open (os.path.join(save_path,'exp_settings.txt'), 'w') as f:
            for key in sorted(args_dict):
                f.write(key+'\t'+str(args_dict[key])+'\n')
        print( save_path)
        #assert False

    locals().update(args_dict)

    if nonlinearity == 'rectify':
        nonlinearity = lasagne.nonlinearities.rectify
    elif nonlinearity == 'gelu':
        nonlinearity = gelu
    
    lbda = np.cast[floatX](args.lbda)
    size = max(10,min(50000,args.size))
    clip_grad = 100
    max_norm = 100

    # SET RANDOM SEED (TODO: rng vs. random.seed)
    if seed is not None:
        np.random.seed(seed)  # for reproducibility
        rng = numpy.random.RandomState(seed)
    else:
        rng = numpy.random.RandomState(np.random.randint(2**32 - 1))
    # --------------------------------------------


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
        
        if convex_combination:
            if init == 'normal':
                h_layer = ConvexBiasLayer(h_layer, b=init.Normal(0.01, 0))
            else: # TODO
                assert False
                h_layer = ConvexBiasLayer(h_layer, b=init.Normal(0.01, 0))
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
            layer = stochasticDenseLayer2([layer,w_layer],num_param, nonlinearity=nonlinearity)
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
            layer = lasagne.layers.DenseLayer(layer, ws[1], nonlinearity=nonlinearity)
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
    predict_probs = theano.function([input_var],y)

    def MCpred(X, inds):
        from utils import MCpred
        return MCpred(X, predict_probs_fn=predict_probs, num_samples=num_samples, inds=inds, returns='preds')



    ##################
    # TRAIN
    X, Y = train_x[:size],train_y[:size]
    Xt, Yt = valid_x,valid_y
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
    records={}
    records['loss'] = []
    records['acc'] = []
    records['val_acc'] = []
    
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
            if i == 0:# or t>8000:
                print 'time', time.time() - t0
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)
                tr_inds = np.random.choice(len(X), num_examples, replace=False)
                te_inds = np.random.choice(len(Xt), num_examples, replace=False)
                tr_acc = (MCpred(X, inds=tr_inds)==Y[tr_inds].argmax(1)).mean()
                te_acc = (MCpred(Xt, inds=te_inds)==Yt[te_inds].argmax(1)).mean()
                assert False
                print '\ttrain acc: {}'.format(tr_acc)
                print '\ttest acc: {}'.format(te_acc)
                records['loss'].append(loss)
                records['acc'].append(tr_acc)
                records['val_acc'].append(te_acc)
                if save_path is not None:
                    np.save(save_path + '_records.npy', records)
                    np.save(save_path + '_params.npy', lasagne.layers.get_all_param_values([h_layer, layer]))
                    if records['val_acc'][-1] == np.max(records['val_acc']):
                        np.save(save_path + '_params_best.npy', lasagne.layers.get_all_param_values([h_layer, layer]))

            t+=1

        
# --------------------------------------------
# Anomaly Detection Metrics
if 1:


    pass

    

