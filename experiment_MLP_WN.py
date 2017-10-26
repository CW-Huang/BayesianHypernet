#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:49:51 2017

@author: Chin-Wei
"""

from BHNs import MLPWeightNorm_BHN
#from concrete_dropout import MLPConcreteDropout_BHN
from ops import load_mnist
from utils import log_normal, log_laplace, train_model, evaluate_model
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
    
class MCdropout_MLP(object):

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
            if j!=len(self.weight_shapes)-1:
                layer = lasagne.layers.dropout(layer)
        
        layer.nonlinearity = lasagne.nonlinearities.softmax
        self.input_var = T.matrix('input_var')
        self.target_var = T.matrix('target_var')
        self.learning_rate = T.scalar('leanring_rate')
        
        self.layer = layer
        self.y = T.clip(lasagne.layers.get_output(layer,self.input_var),
                        0.001, 0.999)
        self.y_unclipped = lasagne.layers.get_output(layer,self.input_var)
        
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


    
if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--lrdecay',default=0,type=int)      
    parser.add_argument('--lr0',default=0.0001,type=float)  
    parser.add_argument('--coupling',default=0,type=int) 
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--size',default=2000,type=int)      
    parser.add_argument('--bs',default=32,type=int)  
    parser.add_argument('--epochs',default=10,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--model',default='BHN_MLPWN',type=str, 
                        choices=['BHN_MLPWN', 'BHN_MLPCD', 'MCdropout_MLP','MCdropout_MLP_NC']) 
                        # TODO: concrete dropout
    parser.add_argument('--anneal',default=0,type=int)
    parser.add_argument('--n_hiddens',default=1,type=int)
    parser.add_argument('--n_units',default=200,type=int)
    parser.add_argument('--totrain',default=1,type=int)
    parser.add_argument('--adv_eval',default=1,type=int)
    parser.add_argument('--avg',default=1,type=int)
    parser.add_argument('--seed',default=427,type=int)
    parser.add_argument('--override',default=1,type=int)
    parser.add_argument('--reinit',default=1,type=int)
    parser.add_argument('--flow',default='RealNVP',type=str, 
                        choices=['RealNVP', 'IAF'])
    parser.add_argument('--alpha',default=2, type=float)
    parser.add_argument('--beta',default=1, type=float)
    parser.add_argument('--save_dir',default='./models',type=str)
    
    
    args = parser.parse_args()
    print args
    
    
    set_rng(np.random.RandomState(args.seed))
    np.random.seed(args.seed+1000)

    
    if args.prior == 'log_normal':
        pr = 0
    if args.prior == 'log_laplace':
        pr = 1
    
    
    if args.model == 'BHN_MLPCD':
        md = 2
    if args.model == 'BHN_MLPWN':
        md = 0
    if args.model == 'MCdropout_MLP':
        md = 1
    if args.model == 'MCdropout_MLP_NC':
        md = 3
    
    
    path = args.save_dir
    if not os.path.exists(path):
        os.makedirs(path)

    name = '{}/mnistWN_md{}nh{}nu{}c{}pr{}lbda{}lr0{}lrd{}an{}s{}seed{}reinit{}alpha{}beta{}flow{}'.format(
        path,
        md,
        args.n_hiddens,
        args.n_units,
        args.coupling,
        pr,
        args.lbda,
        args.lr0,
        args.lrdecay,
        args.anneal,
        args.size,
        args.seed,
        args.reinit,
        args.alpha,
        args.beta,
        args.flow
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
    elif os.path.isfile(os.path.join(os.environ['DATA_PATH'], 'mnist.pkl.gz')):
        filename = os.path.join(os.environ['DATA_PATH'], 'mnist.pkl.gz')
    else:        
        print '\n\tdownloading mnist'
        import download_datasets.mnist
        filename = r'./data/mnist.pkl.gz'

    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
    
    if args.reinit:
        init_batch_size = min(64, size)
        init_batch = train_x[:size][-init_batch_size:].reshape(init_batch_size,784)
    else:
        init_batch = None
        
    if args.model == 'BHN_MLPWN':
        model = MLPWeightNorm_BHN(lbda=lbda,
                                  perdatapoint=perdatapoint,
                                  srng = RandomStreams(seed=args.seed+2000),
                                  prior=prior,
                                  coupling=coupling,
                                  n_hiddens=n_hiddens,
                                  n_units=n_units,
                                  flow=args.flow,
                                  init_batch=init_batch)
    elif args.model == 'BHN_MLPCD':
        model = MLPConcreteDropout_BHN(lbda=lbda,
                                  alpha=args.alpha,
                                  beta=args.beta,
                                  perdatapoint=perdatapoint,
                                  srng = RandomStreams(seed=args.seed+2000),
                                  prior=prior,
                                  coupling=coupling,
                                  n_hiddens=n_hiddens,
                                  n_units=n_units,
                                  flow=args.flow,
                                  init_batch=init_batch)
    elif args.model == 'MCdropout_MLP':
        model = MCdropout_MLP(n_hiddens=n_hiddens,
                              n_units=n_units)
    elif args.model == 'MCdropout_MLP_NC':
        model = MCdropout_MLP(n_hiddens=n_hiddens,
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
                    lr0,lrdecay,bs,epochs,anneal,name,
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
        results = adv_evaluate(test_x,
                               test_y,
                               model.predict_proba,
                               model.input_var,
                               model.target_var,
                               model.y_unclipped,
                               avg=args.avg)
        
        np.save(name+'_adv',results)
        
