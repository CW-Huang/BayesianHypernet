#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:49:51 2017

@author: Chin-Wei
"""

#from concrete_dropout import MLPConcreteDropout_BHN
from ops import load_mnist
from utils import train_model, evaluate_model
import numpy as np

import lasagne
import theano
import theano.tensor as T
import os
from lasagne.random import set_rng
from theano.tensor.shared_randomstreams import RandomStreams
from FGS_eval import evaluate as adv_evaluate


# TODO: add all LCS

lrdefault = 1e-3    
    
class MLP(object):

    def __init__(self,n_hiddens,n_units, n_inputs=784,
                 clip_output=True):
        
        layer = lasagne.layers.InputLayer([None,n_inputs])
        
        self.n_hiddens = n_hiddens
        self.n_units = n_units
        self.weight_shapes = list()        
        self.weight_shapes.append((n_inputs,n_units))
        for i in range(1,n_hiddens):
            self.weight_shapes.append((n_units,n_units))
        self.weight_shapes.append((n_units,10))
        self.num_params = sum(ws[1] for ws in self.weight_shapes)
        
        
        for j,ws in enumerate(self.weight_shapes):
            layer = lasagne.layers.DenseLayer(
                layer,ws[1],
                nonlinearity=lasagne.nonlinearities.rectify
            )

        layer.nonlinearity = lasagne.nonlinearities.softmax
        self.input_var = T.matrix('input_var')
        self.target_var = T.matrix('target_var')
        self.learning_rate = T.scalar('leanring_rate')
        
        self.layer = layer
        if clip_output:
            self.y = T.clip(lasagne.layers.get_output(layer,self.input_var),
                            0.001, 0.999)
        else:
            self.y = lasagne.layers.get_output(layer,self.input_var)
            
        self.y_det = lasagne.layers.get_output(layer,self.input_var,
                                               deterministic=True)
        self.output_var = self.y # aliasing

        losses = lasagne.objectives.categorical_crossentropy(self.y,
                                                             self.target_var)
        self.loss = losses.mean()
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


        

               
def fgm_grad(x, prediction, y):
    loss = T.nnet.categorical_crossentropy(prediction,y)    
    grad = T.grad(loss.mean(), x)
    return theano.function([x,y],grad)
    

def evaluate_save(X,Y,predict_proba,
                  input_var,target_var,prediction,
                  eps=[0,0.001,0.002,0.003,0.004,0.005,0.008,0.01,0.012,0.015,
                       0.02,0.025,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3,0.5],
                  max_n=100,n_classes=10,
                  save_dir='.'):
    
    print 'compiling attacker ...'

    grad = fgm_grad(input_var,prediction,target_var)
    
    def att(x,y,ep):
        grads = grad(x,y)
        signed = np.sign(grads)
        return x + ep * signed
    
    
    N = X.shape[0]
    num_batches = np.ceil(N / float(max_n)).astype(int)
    
    def per_ep(ep):
        Xa = np.zeros(X.shape,dtype='float32')
        for j in range(num_batches):
            x = X[j*max_n:(j+1)*max_n]
            y = Y[j*max_n:(j+1)*max_n]
            xa = att(x,y,ep)
            Xa[j*max_n:(j+1)*max_n] = xa
        
        pred = np.zeros((X.shape[0],n_classes),dtype='float32')    
        for j in range(num_batches):
            x = Xa[j*max_n:(j+1)*max_n]
            pred[j*max_n:(j+1)*max_n] = predict_proba(x)
    
        Y_proba = pred
        
        # generalization
        Y_pred = Y_proba.argmax(-1)
        Y_true = Y.argmax(-1)
        corr = np.equal(Y_pred,Y_true)
        return ep, corr.mean(), Xa
    
    for ep in eps:
        
        ep, corr, Xa = per_ep(ep)
        
        print ep, corr
        np.save('{}/{}'.format(save_dir,ep),Xa)
        
        
        
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lrdecay',default=0,type=int)      
    parser.add_argument('--lr0',default=0.0001,type=float)  
    parser.add_argument('--size',default=2000,type=int)      
    parser.add_argument('--bs',default=32,type=int)  
    parser.add_argument('--epochs',default=10,type=int)
    parser.add_argument('--model',default='MLP',type=str, 
                        choices=['MLP','MLP_NC']) 
                        # TODO: concrete dropout
    parser.add_argument('--n_hiddens',default=1,type=int)
    parser.add_argument('--n_units',default=200,type=int)
    parser.add_argument('--totrain',default=1,type=int)
    parser.add_argument('--adv_eval',default=1,type=int)
    parser.add_argument('--seed',default=427,type=int)
    parser.add_argument('--override',default=1,type=int)
    parser.add_argument('--save_dir',default='./models_MLP',type=str)
    
    
    args = parser.parse_args()
    print args
    
    
    set_rng(np.random.RandomState(args.seed))
    np.random.seed(args.seed+1000)

    
    
    if args.model == 'MLP':
        md = 1
    if args.model == 'MLP_NC':
        md = 2
    
    
    path = args.save_dir
    if not os.path.exists(path):
        os.makedirs(path)

    name = '{}/mnistWN_md{}nh{}nu{}lr0{}lrd{}s{}seed{}'.format(
        path,
        md,
        args.n_hiddens,
        args.n_units,
        args.lr0,
        args.lrdecay,
        args.size,
        args.seed,
    )

    lrdecay = args.lrdecay
    lr0 = args.lr0
    bs = args.bs
    epochs = args.epochs
    n_hiddens = args.n_hiddens
    n_units = args.n_units
    size = max(10,min(50000,args.size))
    
    if os.path.isfile('/data/lisa/data/mnist.pkl.gz'):
        filename = '/data/lisa/data/mnist.pkl.gz'
    elif os.path.isfile(r'./data/mnist.pkl.gz'):
        filename = r'./data/mnist.pkl.gz'
    elif os.path.isfile(os.path.join(os.environ['DATA_PATH'], 'mnist.pkl.gz')):
        filename = os.path.join(os.environ['DATA_PATH'], 'mnist.pkl.gz')
    else:        
        print '\n\tdownloading mnist'
        import download_datasets.mnist
        filename = r'./data/mnist.pkl.gz'

    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
    
        
    
    if args.model == 'MLP':
        model = MLP(n_hiddens=n_hiddens,
                    n_units=n_units)
    elif args.model == 'MLP_NC':
        model = MLP(n_hiddens=n_hiddens,
                    n_units=n_units,clip_output=False)
    else:
        raise Exception('no model named `{}`'.format(args.model))

    va_rec_name = name+'_recs'
    tr_rec_name = name+'_recs_train' # TODO (we're already saving the valid_recs!)
    save_path = name + '.params.npy'
    if os.path.isfile(save_path) and not args.override:
        print 'load best model'
        e0 = model.load(save_path)
        va_recs = open(va_rec_name,'r').read().split('\n')[:e0]
        #tr_recs = open(tr_rec_name,'r').read().split('\n')[:e0]
        rec = max([float(r) for r in va_recs])
        
    else:
        e0 = 0
        rec = 0


    if args.totrain:
        print '\nstart training from epoch {}'.format(e0)
        train_model(model,
                    train_x[:size],train_y[:size],
                    valid_x,valid_y,
                    lr0,lrdecay,bs,epochs,0,name,
                    e0,rec,print_every=999999)
    else:
        print '\nno training'
    
    tr_acc = evaluate_model(model.predict_proba,
                            train_x[:size],train_y[:size])
    print 'train acc: {}'.format(tr_acc)
                   
    va_acc = evaluate_model(model.predict_proba,
                            valid_x,valid_y,n_mc=200)
    print 'valid acc: {}'.format(va_acc)
    
    te_acc = evaluate_model(model.predict_proba,
                            test_x,test_y,n_mc=200)
    print 'test acc: {}'.format(te_acc)


    if args.totrain == 1:
        # report the best valid-model's test acc
        e0 = model.load(save_path)
        te_acc = evaluate_model(model.predict_proba,
                                test_x,test_y,n_mc=200)
        print 'test acc (best valid): {}'.format(te_acc)

        
    if args.adv_eval == 1:
        results = evaluate_save(test_x,
                                test_y,
                                model.predict_proba,
                                model.input_var,
                                model.target_var,
                                model.y,
                                save_dir=path)
        
        np.save(name+'_adv',results)
        
