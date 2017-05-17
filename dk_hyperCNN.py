#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:49:51 2017

@author: Chin-Wei
"""

from BHNs import MLPWeightNorm_BHN
from ops import load_mnist, load_cifar10
#from utils import log_normal, log_laplace
import numpy
import numpy as np

# utils issues...
def log_normal(x,mean,log_var,eps=0.0):
    c = - 0.5 * T.log(2*np.pi)
    return c - log_var/2. - (x - mean)**2 / (2. * T.exp(log_var) + eps)

import lasagne
import theano
import theano.tensor as T
floatX = theano.config.floatX

# DK
from BHNs import HyperCNN
    
class MCdropoutCNN(object):
                     
    def __init__(self, dropout=None, 
            opt='adam',
            dataset='mnist'):
        if dataset == 'mnist':
            weight_shapes = [(32,1,3,3),        # -> (None, 16, 14, 14)
                             (32,32,3,3),       # -> (None, 16,  7,  7)
                             (32,32,3,3)]       # -> (None, 16,  4,  4)
        elif dataset == 'cifar10':
            weight_shapes = [(32,3,5,5),        # -> (None, 16, 16, 16)
                             (32,32,5,5),       # -> (None, 16,  8,  8)
                             (32,32,5,5)]       # -> (None, 16,  4,  4)
        n_kernels = np.array(weight_shapes)[:,1].sum()
        kernel_shape = weight_shapes[0][:1]+weight_shapes[0][2:]
        
        # needs to be consistent with weight_shapes
        args = [32,3,2,'same',lasagne.nonlinearities.rectify]#
        num_filters, filter_size, stride, pad, nonlinearity = args
        self.__dict__.update(locals())
        ##################
        
        if dataset == 'mnist':
            layer = lasagne.layers.InputLayer([None,1,28,28])
        elif dataset == 'cifar10':
            layer = lasagne.layers.InputLayer([None,3,32,32])
        
        for j,ws in enumerate(self.weight_shapes):
            num_filters = ws[1]
            layer = lasagne.layers.Conv2DLayer(layer,
                num_filters, filter_size, stride, pad, nonlinearity
            )
            if dropout is not None and j!=len(self.weight_shapes)-1:
                if dropout == 'spatial':
                    layer = lasagne.layers.spatial_dropout(layer, dropout)
                else:
                    layer = lasagne.layers.dropout(layer, dropout)

        # MLP layers
        layer = lasagne.layers.DenseLayer(layer, 128)
        if dropout is not None and j!=len(self.weight_shapes)-1:
            layer = lasagne.layers.dropout(layer, dropout)
        layer = lasagne.layers.DenseLayer(layer, 10)

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
        if opt == 'adam':
            self.updates = lasagne.updates.adam(self.loss,self.params,
                                                self.learning_rate)
        elif opt == 'momentum':
            self.updates = lasagne.updates.nesterov_momentum(self.loss,self.params,
                                                self.learning_rate)
        elif opt == 'sgd':
            self.updates = lasagne.updates.sgd(self.loss,self.params,
                                                self.learning_rate)

        print '\tgetting train_func'
        self.train_func = theano.function([self.input_var,
                                           self.target_var,
                                           self.dataset_size,
                                           self.learning_rate],
                                          self.loss,
                                          updates=self.updates)
        
        print '\tgetting useful_funcs'
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y_det.argmax(1))
        




# FIXME: I just realize I broke this for all the non-Hyper models... :/ 
def train_model(model,
                X,Y,Xt,Yt,
                lr0=0.1,lrdecay=1,bs=20,epochs=50,
                save_path=None):

    predict_func = model.predict
    
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
    records={}
    records['loss'] = []
    #records['val_loss'] = []
    records['acc'] = []
    records['val_acc'] = []
    records['logpyx'] = []
    records['logpw'] = []
    records['logqw'] = []
    records['logpyx_grad'] = []
    records['logpw_grad'] = []
    records['logqw_grad'] = []
    
    t = 0
    for e in range(epochs):
        
        if lrdecay:
            lr = lr0 * 10**(-e/float(epochs-1))
        else:
            lr = lr0         
            
        for i in range(N/bs):
            x = X[i*bs:(i+1)*bs]
            y = Y[i*bs:(i+1)*bs]
            
            loss = model.train_func(x,y,N,lr)
            
            if i == 0:# or t>8000:
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)
                tr_acc = (predict_func(X)==Y.argmax(1)).mean()
                te_acc = (predict_func(Xt)==Yt.argmax(1)).mean()
                print '\ttrain acc: {}'.format(tr_acc)
                print '\ttest acc: {}'.format(te_acc)
                records['loss'].append(loss)
                records['acc'].append(tr_acc)
                records['val_acc'].append(te_acc)
                monitored = model.monitor_func(x,y,N,lr)
                # why do I need the mean()???
                monitored = [mm.mean().item() for mm in monitored]
                print "logpyx, logpw, logqw", monitored[0:3]
                print "logpyx_grad, logpw_grad, logqw_grad", monitored[3:]
                records['logpyx'].append(monitored[0])
                records['logpw'].append(monitored[1])
                records['logqw'].append(monitored[2])
                records['logpyx_grad'].append(monitored[3])
                records['logpw_grad'].append(monitored[4])
                records['logqw_grad'].append(monitored[5])
                if all(np.isnan(mm) for mm in monitored):
                    assert False
                if save_path is not None:
                    np.save(save_path, records)
            t+=1
        
    return records


def evaluate_model(predict_proba,X,Y,Xt,Yt,n_mc=1000):
    n = X.shape[0]
    MCt = np.zeros((n_mc,X.shape[0],10))
    MCv = np.zeros((n_mc,Xt.shape[0],10))
    for i in range(n_mc):
        MCt[i] = predict_proba(X)
        MCv[i] = predict_proba(Xt)
    
    Y_pred = MCt.mean(0).argmax(-1)
    Y_true = Y.argmax(-1)
    Yt_pred = MCv.mean(0).argmax(-1)
    Yt_true = Yt.argmax(-1)
    
    tr = np.equal(Y_pred,Y_true).mean()
    va = np.equal(Yt_pred,Yt_true).mean()
    print "train perf=", tr
    print "valid perf=", va

    ind_positive = np.arange(Xt.shape[0])[Yt_pred == Yt_true]
    ind_negative = np.arange(Xt.shape[0])[Yt_pred != Yt_true]
    
    ind = ind_negative[-1] #TO-DO: complete evaluation
    for ii in range(15): 
        print np.round(MCv[ii][ind] * 1000)
    
    ind = ind_negative[-2] #TO-DO: complete evaluation
    for ii in range(15): 
        print np.round(MCv[ii][ind] * 1000)

#def main():
if __name__ == '__main__':
    

    import os
    import sys
    import argparse

    # ALPHABETIC ORDER
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs',default=128,type=int)  
    parser.add_argument('--coupling',default=4,type=int) 
    parser.add_argument('--dataset',default='mnist',type=str)
    parser.add_argument('--epochs',default=100,type=int)
    parser.add_argument('--lr0',default=0.001,type=float)  
    parser.add_argument('--lrdecay',default=0,type=int)  
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--model',default='hyperCNN',type=str)
    parser.add_argument('--opt',default='adam',type=str)
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--size',default=50000,type=int)      
    #
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--verbose', type=int, default=1)
    #locals().update(parser.parse_args().__dict__)


    # ---------------------------------------------------------------
    # PARSE ARGS and SET-UP SAVING (save_path/exp_settings.txt)
    # NTS: we name things after the filename + provided args.  We could also save versions (ala Janos), and/or time-stamp things.
    # TODO: loading

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


    print "\n\n\n-----------------------------------------------------------------------\n\n\n"
    print args
    lbda = np.cast['float32'](lbda)


    # ---------------------------------------------------------------
    # SET RANDOM SEED (TODO: rng vs. random.seed)


    if seed is not None:
        np.random.seed(seed)  # for reproducibility
        rng = numpy.random.RandomState(seed)
    else:
        rng = numpy.random.RandomState(np.random.randint(2**32 - 1))

    # ---------------------------------------------------------------
    # RUN STUFF 

    if prior=='log_normal':
        prior = log_normal
    elif prior=='log_laplace':
        prior = log_laplace
    else:
        raise Exception('no prior named `{}`'.format(prior))
    size = max(10,min(50000,size))
    
    if dataset=='mnist':
        filename = '/data/lisa/data/mnist.pkl.gz'
        train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
        train_x = train_x.reshape((-1, 1, 28, 28))
        valid_x = valid_x.reshape((-1, 1, 28, 28))
        test_x = test_x.reshape((-1, 1, 28, 28))
    elif dataset=='cifar10':
        filename = 'cifar10.pkl'
        train_x, train_y, test_x, test_y = load_cifar10(filename)
        train_x = train_x.reshape((-1, 3, 32, 32))
        test_x = test_x.reshape((-1, 3, 32, 32))
        
        valid_x = test_x.copy()
        valid_y = test_y.copy()
    

    if model == 'hyperCNN':
        model = HyperCNN(lbda=lbda,
                         perdatapoint=perdatapoint,
                         prior=prior,
                         coupling=coupling,
                         dataset=dataset,
                         opt=opt)
    elif model == 'CNN':
        model = MCdropoutCNN(dataset=dataset,opt=opt)
    elif model == 'CNN_spatial_dropout':
        model = MCdropoutCNN(dropout='spatial',
                             dataset=dataset,opt=opt)
    elif model == 'CNN_dropout':
        model = MCdropoutCNN(dropout=1,
                             dataset=dataset,opt=opt)
    else:
        raise Exception('no model named `{}`'.format(model))
        
    recs = train_model(model,
                       train_x[:size],train_y[:size],
                       valid_x,valid_y,
                       lr0,lrdecay,bs,epochs)
    
    from helpers import plot_dict
    plot_dict(recs)
    
    evaluate_model(model.predict_proba,
                   train_x[:size],train_y[:size],
                   valid_x,valid_y)
    
    print '\tevaluating train/test sets'
    evaluate_model(model.predict_proba,
                   train_x[:10000],train_y[:size],
                   test_x,test_y)



