# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:45:40 2017

@author: Chin-Wei
"""


from ops import load_mnist
from utils import log_normal, log_laplace, train_model, evaluate_model
import numpy as np

import lasagne
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
floatX = theano.config.floatX
import os
from lasagne.random import set_rng
from lasagne.objectives import categorical_crossentropy as cc
from lasagne.layers import get_output
from theano.tensor.shared_randomstreams import RandomStreams

from modules import hypernet, N_get_output, get_elbo

# TODO: add all LCS

lrdefault = 1e-3    
    
class MLP(object):

    def __init__(self,
                 n_hiddens, n_units, 
                 n_inputs=784,
                 dropout=False, 
                 flow='IAF', 
                 norm_type='WN',
                 coupling=0, 
                 n_units_h=200, 
                 static_bias=True,
                 prior=log_normal,
                 lbda=1,
                 srng=RandomStreams(seed=427)):
        """
        flow: 
            if None, then just regular MLE estimate of parameters
            flow can be `IAF` or `NVP` to approximate the rescaling 
            parameters (and shift) of Weightnorm or Batchnorm 
        
        coupling: 
            number of transformation layers using `IAF` or `RealNVP` if 
            flow is not None
            
        dropout:
            dropout layer after activation
        
        static_bias:
            if one wants the hyper net to output the shifting parameters
            of WN/BN of flow is not None
        
        """
        
        
        layer = lasagne.layers.InputLayer([None,n_inputs])
        
        self.n_hiddens = n_hiddens
        self.n_units = n_units
        self.weight_shapes = list()        
        self.weight_shapes.append((n_inputs,n_units))
        for i in range(1,n_hiddens):
            self.weight_shapes.append((n_units,n_units))
        self.weight_shapes.append((n_units,10))
        self.num_params = sum(ws[1] for ws in self.weight_shapes)
        self.flow = flow
        self.norm_type = norm_type
        self.coupling = coupling
        self.dropout = dropout
        self.static_bias = static_bias
        self.prior = prior
        self.lbda = lbda
        
        
        for j,ws in enumerate(self.weight_shapes):
            layer = lasagne.layers.DenseLayer(
                layer,ws[1],
                nonlinearity=lasagne.nonlinearities.rectify
            )
            if dropout:
                if j!=len(self.weight_shapes)-1:
                    layer = lasagne.layers.dropout(layer)
        
        layer.nonlinearity = lasagne.nonlinearities.softmax
        self.input_var = T.matrix('input_var')
        self.target_var = T.matrix('target_var')
        self.learning_rate = T.scalar('leanring_rate')
        self.inputs = [self.input_var,
                       self.target_var,
                       self.learning_rate]
        
        self.layer = layer
        if flow is None:
            self.output_var = get_output(layer,self.input_var)
            self.output_var_det = get_output(layer,self.input_var,
                                             deterministic=True)
            
            losses = cc(self.y,self.target_var)
            self.loss = losses.mean()         
            self.prints = []
               
        elif flow == 'IAF' or flow == 'RealNVP':
            
            self.dataset_size = T.scalar('dataset_size')
            self.beta = T.scalar('beta') # anealing weight
            self.inputs = [self.input_var,
                           self.target_var,
                           self.dataset_size,
                           self.learning_rate,
                           self.beta]
            
            copies = 1 if self.static_bias else 2
            hnet, ld, num_params = hypernet(layer, 
                                            n_units_h, 
                                            coupling, 
                                            flow,
                                            copies=copies)
            static_bias = theano.shared(
                np.zeros((num_params)).astype(floatX)
            ) if self.static_bias else None
            
            ep = srng.normal(size=(1,num_params),dtype=floatX)        
            
   
            output_var = N_get_output(layer,
                                      self.input_var,hnet,ep,
                                      norm_type=norm_type,
                                      static_bias=static_bias)
            weights = get_output(hnet,ep)
            logdets = get_output(ld,ep)
            
            self.num_params = num_params
            self.N_bias = static_bias
            self.hnet = hnet
            self.ep = ep
            self.output_var = output_var
            self.weights = weights
            self.logdets = logdets
            
            loss, prints = get_elbo(self.output_var,
                                    self.target_var,
                                    self.weights,
                                    self.logdets,
                                    self.beta,
                                    self.dataset_size,
                                    prior=self.prior,
                                    lbda=self.lbda,
                                    output_type='categorical')
            self.loss = loss
            self.prints = prints
            
        
        
        self.params = lasagne.layers.get_all_params(self.layer) + \
                      lasagne.layers.get_all_params(self.hnet)
        if hasattr(self,'N_bias'):
            if self.N_bias is not None:
                self.params.append(self.N_bias)
            
        self.updates = lasagne.updates.adam(self.loss,self.params,
                                            self.learning_rate)

        print '\tgetting train_func'
        if len(self.inputs) == 3:
            self.train_func_ = theano.function(self.inputs,
                                               [self.loss,]+self.prints,
                                               updates=self.updates)        
            self.tran_func = lambda x,y,n,lr,w: self.train_func_(x,y,lr)
        elif len(self.inputs) == 5:
            self.train_func = theano.function(self.inputs,
                                              [self.loss,]+self.prints,
                                              updates=self.updates)
                                               
        print '\tgetting useful_funcs'
        self.predict_proba = theano.function([self.input_var],self.output_var)
        
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
    parser.add_argument('--dropout',action='store_true',
                        help='dropout applied to post-activation')
    parser.add_argument('--anneal',default=0,type=int)
    parser.add_argument('--n_hiddens',default=1,type=int)
    parser.add_argument('--n_units',default=200,type=int)
    parser.add_argument('--totrain',default=1,type=int)
    parser.add_argument('--seed',default=427,type=int)
    parser.add_argument('--override',default=1,type=int)
    parser.add_argument('--reinit',default=1,type=int)
    parser.add_argument('--flow',default='RealNVP',type=str, 
                        choices=['RealNVP', 'IAF'])
    parser.add_argument('--n_units_h',default=200, type=int)
    parser.add_argument('--norm_type',default='WN', type=str)
    parser.add_argument('--static_bias',default=1,type=int)
    parser.add_argument('--alpha',default=2, type=float)
    parser.add_argument('--beta',default=1, type=float)
    parser.add_argument('--save_dir',default='./models',type=str)
    
    
    args = parser.parse_args()
    print args
    
    if args.flow == '0':
        args.flow = None
    elif args.flow == 'IAF' or args.flow == 'RealNVP':
        pass
    else:
        raise Exception('flow type {} not supported'.format(args.flow))
    
    
    set_rng(np.random.RandomState(args.seed))
    np.random.seed(args.seed+1000)

    
    if args.prior == 'log_normal':
        pr = 0
    if args.prior == 'log_laplace':
        pr = 1
    
    

    
    if args.dropout:
        dp = 1
    else:
        dp = 0
        
    if args.flow is None:
        fl = '0'
    else:
        fl = args.flow
    
    if args.static_bias:
        sb = 1
    else:
        sb = 0
    
    path = args.save_dir
    if not os.path.exists(path):
        os.makedirs(path)

    name = '{}/MLP_nh{}nu{}flow{}{}sb{}c{}pr{}lbda{}lr0{}lrd{}an{}s{}seed{}dp{}'.format(
        path,
        args.n_hiddens,
        args.n_units,
        fl,
        args.n_units_h,
        args.static_bias,
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
        dp
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
    n_inputs = 784
    
    


    model = MLP(n_hiddens, n_units, 
                 n_inputs=n_inputs,
                 dropout=args.dropout, 
                 flow=args.flow, 
                 norm_type=args.norm_type,
                 coupling=coupling, 
                 n_units_h=args.n_units_h, 
                 static_bias=args.static_bias,
                 prior=prior,
                 lbda=lbda,
                 srng=RandomStreams(seed=427))
    
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
                    e0,rec)
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


