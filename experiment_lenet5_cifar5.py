# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:49:51 2017

@author: Chin-Wei
"""


from ops import load_mnist, load_cifar10, load_cifar5
from utils import log_normal, log_laplace, train_model, evaluate_model
import numpy as np

import lasagne
import theano
import theano.tensor as T
floatX = theano.config.floatX

# DK / CW
from BHNs import HyperWN_CNN
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne import nonlinearities
from lasagne.random import set_rng
import os
rectify = nonlinearities.rectify

class MCdropoutCNN(object):
                     
    def __init__(self, dropout=None, 
                 input_channels=3,
                 input_shape = (3,32,32),
                 n_convlayers=2,
                 n_channels = 192,
                 stride = 1,
                 pad = 'valid',
                 nonl = rectify,
                 dataset='mnist', 
                 n_mlplayers=1,
                 n_units=1000,
                 n_classes=10):

        weight_shapes = list()
        args = list()
        n_channels = n_channels if isinstance(n_channels,list) else \
                     [n_channels for i in range(n_convlayers)]
        in_chan = input_channels
        for i in range(n_convlayers):
            out_chan = n_channels[i]
            weight_shape = (out_chan, in_chan, kernel_size, kernel_size)
            weight_shapes.append(weight_shape)
            
            num_filters = out_chan
            filter_size = kernel_size
            stride = stride
            pad = pad
            nonl = nonl
            # pool every `pool` conv layers
            if (i+1)%pool_per == 0:
                pool = 'max'
            else:
                pool = None
            arg = (num_filters,filter_size,stride,pad,nonl,pool)
            args.append(arg)
            in_chan = out_chan
        
        num_hids = n_units
        
        
        n_kernels = np.array(weight_shapes)[:,1].sum()
        kernel_shape = weight_shapes[0][:1]+weight_shapes[0][2:]
        
        # needs to be consistent with weight_shapes
        
        
        self.__dict__.update(locals())
        ##################
        
        layer = lasagne.layers.InputLayer((None,)+input_shape)
        
        for j,(ws,arg) in enumerate(zip(weight_shapes,args)):
            num_filters = ws[1]
            num_filters, filter_size, stride, pad, nonlinearity, pool = arg
            layer = lasagne.layers.Conv2DLayer(layer,
                num_filters, filter_size, stride, pad, nonlinearity
            )
            if dropout is not None and j!=len(weight_shapes)-1:
                if dropout == 'spatial':
                    layer = lasagne.layers.spatial_dropout(layer, 0.5)
                else:
                    layer = lasagne.layers.dropout(layer, 0.5)
            
            if pool=='max':
                layer = lasagne.layers.MaxPool2DLayer(layer,2)

            print layer.output_shape
        # MLP layers
        for i in range(n_mlplayers):
            layer = lasagne.layers.DenseLayer(layer, num_hids)
            if dropout is not None and i!=n_mlplayers-1:
                layer = lasagne.layers.dropout(layer, 0.5)
        layer = lasagne.layers.DenseLayer(layer, n_classes)

        layer.nonlinearity = lasagne.nonlinearities.softmax
        self.input_var = T.tensor4('input_var')
        self.target_var = T.matrix('target_var')
        self.learning_rate = T.scalar('leanring_rate')
        self.dataset_size = T.scalar('dataset_size') # useless
        
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
                                            self.dataset_size,
                                            self.learning_rate],
                                           self.loss,
                                           updates=self.updates)
        
        self.train_func = lambda a,b,c,d,w: self.train_func_(a,b,c,d)
        print '\tgetting useful_funcs'
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y_det.argmax(1))
        


if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset',default='cifar5',type=str)
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--lrdecay',default=0,type=int)      
    parser.add_argument('--lr0',default=0.0001,type=float)  
    parser.add_argument('--coupling',default=0,type=int) 
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--size',default=2000,type=int)      
    parser.add_argument('--bs',default=32,type=int)  
    parser.add_argument('--epochs',default=10,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--model',default='HyperWN_CNN',type=str) 
    parser.add_argument('--anneal',default=0,type=int)
    parser.add_argument('--n_hiddens',default=1,type=int)
    parser.add_argument('--n_units',default=200,type=int)
    parser.add_argument('--totrain',default=1,type=int)
    parser.add_argument('--seed',default=427,type=int)
    parser.add_argument('--override',default=1,type=int)
    parser.add_argument('--reinit',default=1,type=int)
    parser.add_argument('--flow',default='RealNVP',type=str, 
                        choices=['RealNVP', 'IAF'])
    parser.add_argument('--alpha',default=2, type=float)
    parser.add_argument('--beta',default=1, type=float)
    parser.add_argument('--save_dir',default='./models_CNN',type=str)
    
    
    args = parser.parse_args()
    print args
    
    
    set_rng(np.random.RandomState(args.seed))
    np.random.seed(args.seed+1000)

    
    if args.prior == 'log_normal':
        pr = 0
    if args.prior == 'log_laplace':
        pr = 1
    
    pr = {'log_normal':0,
          'log_laplace':1}[args.prior]
    
    md = {'HyperWN_CNN':0, 
          'CNN':1,
          'CNN_spatial_dropout':2,
          'CNN_dropout':3}[args.model]
          
    ds = {'mnist':0,
          'cifar5':1,
          'cifar10':2}[args.dataset]
    
    path = args.save_dir
    if not os.path.exists(path):
        os.makedirs(path)

    name = '{}/WNCNN_md{}ds{}nh{}nu{}c{}pr{}lbda{}lr0{}lrd{}an{}s{}seed{}' \
           'reinit{}alpha{}beta{}flow{}'.format(
        path,
        md,
        ds,
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
    dataset = args.dataset
    anneal = args.anneal
    if args.prior=='log_normal':
        prior = log_normal
    elif args.prior=='log_laplace':
        prior = log_laplace
    else:
        raise Exception('no prior named `{}`'.format(args.prior))
    size = max(10,min(50000,args.size))
    
    print '\tloading dataset'
    if 0:
        if dataset=='mnist':
            filename = '/data/lisa/data/mnist.pkl.gz'
            train_x, train_y, valid_x, valid_y, test_x, test_y = \
                load_mnist(filename)
            train_x = train_x.reshape((-1, 1, 28, 28))
            valid_x = valid_x.reshape((-1, 1, 28, 28))
            test_x = test_x.reshape((-1, 1, 28, 28))
            input_channels = 1
            input_shape = (1,28,28)
            n_classes = 10
            n_convlayers = 2
            n_channels = [20,50]
            kernel_size = 5
            n_mlplayers = 1
            n_units = 500
            stride = 1
            pad = 'valid'
            nonl = rectify
            pool_per = 1
        elif dataset=='cifar10':
            filename = 'cifar10.pkl'
            train_x, train_y, valid_x, valid_y, test_x, test_y = \
                load_cifar10(filename,seed=args.seed)
            train_x = train_x.reshape((-1, 3, 32, 32))
            test_x = test_x.reshape((-1, 3, 32, 32))
            
            input_channels = 3
            input_shape = (3,32,32)
            n_classes = 10
            n_convlayers = 4
            n_channels = 192
            kernel_size = 5
            n_mlplayers = 1
            n_units = 1000
            stride = 1
            pad = 'valid'
            nonl = rectify
            pool_per = 2
        elif dataset=='cifar5':
            filename = 'cifar10.pkl'
            train_x, train_y, valid_x, valid_y, test_x, test_y = \
                load_cifar5(filename,seed=args.seed)
            train_x = train_x.reshape((-1, 3, 32, 32))
            test_x = test_x.reshape((-1, 3, 32, 32))
            
            input_channels = 3
            input_shape = (3,32,32)
            n_classes = 5 
            n_convlayers = 2
            n_channels = 192
            kernel_size = 5
            n_mlplayers = 1
            n_units = 1000
            stride = 1
            pad = 'valid'
            nonl = rectify
            pool_per = 1
            
    if args.model == 'HyperWN_CNN':
        model = HyperWN_CNN(lbda=lbda,
                            perdatapoint=perdatapoint,
                            srng=RandomStreams(seed=427),
                            prior=prior,
                            coupling=coupling,
                            input_channels=input_channels,
                            input_shape=input_shape,
                            n_classes=n_classes,
                            n_convlayers=n_convlayers,
                            n_channels=n_channels,
                            kernel_size=kernel_size,
                            n_mlplayers=n_mlplayers,
                            n_units=n_units,
                            stride=stride,
                            pad=pad,
                            nonl=nonl,
                            pool_per=pool_per)
    elif args.model == 'CNN':
        model = MCdropoutCNN(dataset=dataset,n_classes=n_classes)
    elif args.model == 'CNN_spatial_dropout':
        model = MCdropoutCNN(dropout='spatial',
                             dataset=dataset,n_classes=n_classes)
    elif args.model == 'CNN_dropout':
        model = MCdropoutCNN(dropout=1,
                             dataset=dataset,n_classes=n_classes)
    else:
        raise Exception('no model named `{}`'.format(args.model))
    

    va_rec_name = name+'_recs'
    save_path = name + '.params.npy'
    if os.path.isfile(save_path) and not args.override:
        print 'load best model'
        e0 = model.load(save_path)
        va_recs = open(va_rec_name,'r').read().split('\n')[:e0]
        rec = max([float(r) for r in va_recs])
        
    else:
        e0 = 0
        rec = 0


    if args.to_train:
        print '\tbegin training'
        train_model(model,
                    train_x[:size],train_y[:size],
                    valid_x,valid_y,
                    lr0,lrdecay,bs,epochs,anneal,name,e0,rec,
                    print_every=10,
                    n_classes=n_classes)
    

                    
    print '\tevaluating train/valid sets'
    evaluate_model(model.predict_proba,
                   train_x[:min(size,10000)],train_y[:min(size,10000)],
                   valid_x,valid_y,
                   n_classes=n_classes)
    
    print '\tevaluating train/test sets'
    evaluate_model(model.predict_proba,
                   train_x[:min(size,10000)],train_y[:min(size,10000)],
                   test_x,test_y,
                   n_classes=n_classes)

    if args.totrain == 1:
        # report the best valid-model's test acc
        e0 = model.load(save_path)
        te_acc = evaluate_model(model.predict_proba,
                                test_x,test_y,n_mc=200)
        print 'test acc (best valid): {}'.format(te_acc)

    


