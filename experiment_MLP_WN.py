#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:49:51 2017

@author: Chin-Wei
"""

from BHNs import MLPWeightNorm_BHN
from ops import load_mnist
from utils import log_normal, log_laplace
import numpy as np

import lasagne
import theano
import theano.tensor as T
import os
from lasagne.random import set_rng
from theano.tensor.shared_randomstreams import RandomStreams


lrdefault = 1e-3    
    
class MCdropout_MLP(object):

    def __init__(self,n_hiddens,n_units):
        
        layer = lasagne.layers.InputLayer([None,784])
        
        self.n_hiddens = n_hiddens
        self.n_units = n_units
        self.weight_shapes = list()        
        self.weight_shapes.append((784,n_units))
        for i in range(1,n_hiddens):
            self.weight_shapes.append((n_units,n_units))
        self.weight_shapes.append((n_units,10))
        self.num_params = sum(ws[1] for ws in self.weight_shapes)
        
        
        for j,ws in enumerate(self.weight_shapes):
            layer = lasagne.layers.DenseLayer(
                layer,ws[1],
                nonlinearity=lasagne.nonlinearities.rectify
            )
            if j!=len(self.weight_shapes)-1:
                layer = lasagne.layers.dropout(layer)
        
        layer.nonlinearity = lasagne.nonlinearities.softmax
        self.input_var = T.matrix('input_var')
        self.target_var = T.matrix('target_var')
        self.learning_rate = T.scalar('leanring_rate')
        
        self.layer = layer
        self.y = lasagne.layers.get_output(layer,self.input_var)
        self.y_det = lasagne.layers.get_output(layer,self.input_var,
                                               deterministic=True)
        
        losses = lasagne.objectives.categorical_crossentropy(self.y,
                                                             self.target_var)
        self.loss = losses.mean() + self.dataset_size * 0.
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
        
    def train_func(self,x,y,n,lr=lrdefault,w=1.0):
        return self.train_func_(x,y,lr)

    def save(self,save_path):
        np.save(save_path, [p.get_value() for p in self.params])

    def load(self,save_path):
        values = np.load(save_path)

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


def train_model(train_func,predict_func,X,Y,Xt,Yt,
                lr0=0.1,lrdecay=1,bs=20,epochs=50,anneal=0,name='0'):
    
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
    records=[0]
    save_path = name + '.params'
    
    t = 0
    for e in range(epochs):
        
        if lrdecay:
            lr = lr0 * 10**(-e/float(epochs-1))
        else:
            lr = lr0         
        
        if anneal:
            w = min(1.0,0.001+e/(epochs/2.))
        else:
            w = 1.0         
            
        for i in range(N/bs):
            x = X[i*bs:(i+1)*bs]
            y = Y[i*bs:(i+1)*bs]
            
            loss = train_func(x,y,N,lr,w)
            
            if t%100==0:
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)
                tr_acc = (predict_func(X)==Y.argmax(1)).mean()
                te_acc = (predict_func(Xt)==Yt.argmax(1)).mean()
                print '\ttrain acc: {}'.format(tr_acc)
                print '\ttest acc: {}'.format(te_acc)
            t+=1
        
        va_acc = evaluate_model(model.predict_proba,Xt,Yt,20)
        print '\n\nva acc at epochs {}: {}'.format(e,va_acc)    
        
        if va_acc > np.max(records):
            print '.... save best model .... '
            model.save(save_path)
        
        print '\n\n'
        records.append(va_acc)
        
        
        
    return records



def evaluate_model(predict_proba,X,Y,n_mc=100,max_n=100):
    MCt = np.zeros((n_mc,X.shape[0],10))
    
    N = X.shape[0]
    num_batches = np.ceil(N / float(max_n)).astype(int)
    for i in range(n_mc):
        for j in range(num_batches):
            x = X[j*max_n:(j+1)*max_n]
            MCt[i,j*max_n:(j+1)*max_n] = predict_proba(x)
    
    Y_pred = MCt.mean(0).argmax(-1)
    Y_true = Y.argmax(-1)
    return np.equal(Y_pred,Y_true).mean()


    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--lrdecay',default=0,type=int)      
    parser.add_argument('--lr0',default=0.1,type=float)  
    parser.add_argument('--coupling',default=0,type=int) 
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--size',default=10000,type=int)      
    parser.add_argument('--bs',default=20,type=int)  
    parser.add_argument('--epochs',default=5,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--model',default='BHN_MLPWN',type=str)
    parser.add_argument('--anneal',default=0,type=int)
    parser.add_argument('--n_hiddens',default=1,type=int)
    parser.add_argument('--n_units',default=200,type=int)
    parser.add_argument('--totrain',default=1,type=int)
    parser.add_argument('--loadbest',default=0,type=int)
    parser.add_argument('--seed',default=427,type=int)
    
    args = parser.parse_args()
    print args
    
    
    set_rng(np.random.RandomState(args.seed))
    np.random.seed(args.seed+1000)

    
    if args.prior == 'log_normal':
        pr = 0
    if args.prior == 'log_laplace':
        pr = 1
        
    path = 'models'
    name = './{}/mnistWN_nh{}nu{}c{}pr{}lbda{}lr0{}lrd{}an{}s{}seed{}'.format(
        path,
        args.n_hiddens,
        args.n_units,
        args.coupling,
        pr,
        args.lbda,
        args.lr0,
        args.lrdecay,
        args.anneal,
        args.size,
        args.seed
    )

    coupling = args.coupling
    perdatapoint = args.perdatapoint
    lrdecay = args.lrdecay
    lr0 = args.lr0
    lbda = np.cast['float32'](args.lbda)
    bs = args.bs
    epochs = args.epochs
    n_hiddens = args.n_hiddens
    n_units = args.n_units
    anneal = args.anneal
    if args.prior=='log_normal':
        prior = log_normal
    elif args.prior=='log_laplace':
        prior = log_laplace
    else:
        raise Exception('no prior named `{}`'.format(args.prior))
    size = max(10,min(50000,args.size))
    
    if os.path.isfile('/data/lisa/data/mnist.pkl.gz'):
        filename = '/data/lisa/data/mnist.pkl.gz'
    elif os.path.isfile(r'./data/mnist.pkl.gz'):
        filename = r'./data/mnist.pkl.gz'
    else:        
        print '\n\tdownloading mnist'
        import download_datasets.mnist
        filename = r'./data/mnist.pkl.gz'

    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
    
    if args.model == 'BHN_MLPWN':
        model = MLPWeightNorm_BHN(lbda=lbda,
                                  perdatapoint=perdatapoint,
                                  srng = RandomStreams(seed=args.seed+2000),
                                  prior=prior,
                                  coupling=coupling,
                                  n_hiddens=n_hiddens,
                                  n_units=n_units)
    elif args.model == 'MCdropout_MLP':
        model = MCdropout_MLP(n_hiddens=n_hiddens,
                              n_units=n_units)
    else:
        raise Exception('no model named `{}`'.format(args.model))

    if args.loadbest:
        print 'load best model'
        save_path = name + '.params.npy'
        model.load(save_path)

    if args.totrain:
        recs = train_model(model.train_func,model.predict,
                           train_x[:size],train_y[:size],
                           valid_x,valid_y,
                           lr0,lrdecay,bs,epochs,anneal,name)
        np.save(name+'_recs.npy',recs)
    else:
        print 'no training'
    
    tr_acc = evaluate_model(model.predict_proba,
                            train_x[:size],train_y[:size])
    print 'train acc: {}'.format(tr_acc)
                   
    va_acc = evaluate_model(model.predict_proba,
                            valid_x,valid_y)
    print 'valid acc: {}'.format(va_acc)
    
    te_acc = evaluate_model(model.predict_proba,
                            test_x,test_y)
    print 'test acc: {}'.format(te_acc)


