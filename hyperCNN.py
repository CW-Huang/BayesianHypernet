# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:49:51 2017

@author: Chin-Wei
"""


from ops import load_mnist, load_cifar10
from utils import log_normal, log_laplace
import numpy as np

import lasagne
import theano
import theano.tensor as T
floatX = theano.config.floatX

# DK / CW
from BHNs import HyperCNN, Conv2D_BHN
    
class MCdropoutCNN(object):
                     
    def __init__(self, dropout=None, dataset='mnist'):
        if dataset == 'mnist':
            weight_shapes = [(32,1,3,3),        # -> (None, 16, 14, 14)
                             (32,32,3,3),       # -> (None, 16,  7,  7)
                             (32,32,3,3)]       # -> (None, 16,  4,  4)
                             
            args = [[32,3,1,'same',lasagne.nonlinearities.rectify,'max'],
                    [32,3,1,'same',lasagne.nonlinearities.rectify,'max'],
                    [32,3,1,'same',lasagne.nonlinearities.rectify,'max']]
            num_hids = 128
            
        elif dataset == 'cifar10':
            weight_shapes = [(64,3  ,3,3),        
                             (64,128,3,3),      
                             (64,128,3,3),      
                             (64,128,3,3)]
            args = [[64,3,1,'valid',lasagne.nonlinearities.rectify,None ],
                    [64,3,1,'valid',lasagne.nonlinearities.rectify,'max'],
                    [64,3,1,'valid',lasagne.nonlinearities.rectify,None ],
                    [64,3,1,'valid',lasagne.nonlinearities.rectify,'max']]
            num_hids = 512
            
            
        n_kernels = np.array(weight_shapes)[:,1].sum()
        kernel_shape = weight_shapes[0][:1]+weight_shapes[0][2:]
        
        # needs to be consistent with weight_shapes
        
        
        self.__dict__.update(locals())
        ##################
        
        if dataset == 'mnist':
            layer = lasagne.layers.InputLayer([None,1,28,28])
        elif dataset == 'cifar10':
            layer = lasagne.layers.InputLayer([None,3,32,32])
        
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
        layer = lasagne.layers.DenseLayer(layer, num_hids)
        if dropout is not None and j!=len(self.weight_shapes)-1:
            layer = lasagne.layers.dropout(layer, 0.5)
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
        self.updates = lasagne.updates.adam(self.loss,self.params,
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
        



def train_model(train_func,predict_func,X,Y,Xt,Yt,
                lr0=0.1,lrdecay=1,bs=20,epochs=50):
    
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
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
            
            if t%500==0:
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)
                tr_acc = (predict_func(X[:1000])==Y[:1000].argmax(1)).mean()
                te_acc = (predict_func(Xt[:1000])==Yt[:1000].argmax(1)).mean()
                print '\ttrain acc: {}'.format(tr_acc)
                print '\ttest acc: {}'.format(te_acc)
            t+=1
            
        records.append(loss)
        
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
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # boolean: 1 -> True ; 0 -> False 
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--lrdecay',default=0,type=int)  
    
    parser.add_argument('--lr0',default=0.001,type=float)  
    parser.add_argument('--coupling',default=4,type=int) 
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--size',default=10000,type=int)      
    parser.add_argument('--bs',default=20,type=int)  
    parser.add_argument('--epochs',default=50,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--model',default='CNN_dropout',type=str)
    parser.add_argument('--dataset',default='cifar10',type=str)
    
    args = parser.parse_args()
    print args
    
    coupling = args.coupling
    perdatapoint = args.perdatapoint
    lrdecay = args.lrdecay
    lr0 = args.lr0
    lbda = np.cast['float32'](args.lbda)
    bs = args.bs
    epochs = args.epochs
    dataset = args.dataset
    if args.prior=='log_normal':
        prior = log_normal
    elif args.prior=='log_laplace':
        prior = log_laplace
    else:
        raise Exception('no prior named `{}`'.format(args.prior))
    size = max(10,min(50000,args.size))
    
    print '\tloading dataset'
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
    

    if args.model == 'hyperCNN':
        model = HyperCNN(lbda=lbda,
                         perdatapoint=perdatapoint,
                         prior=prior,
                         coupling=coupling,
                         dataset=dataset)
    elif args.model == 'CNN':
        model = MCdropoutCNN(dataset=dataset)
    elif args.model == 'CNN_spatial_dropout':
        model = MCdropoutCNN(dropout='spatial',
                             dataset=dataset)
    elif args.model == 'CNN_dropout':
        model = MCdropoutCNN(dropout=1,
                             dataset=dataset)
    elif args.model == 'CNN_BHN':
        model = Conv2D_BHN(lbda=lbda,
                           perdatapoint=perdatapoint,
                           prior=prior,
                           coupling=coupling,
                           dataset=dataset)
    else:
        raise Exception('no model named `{}`'.format(args.model))
    

    print '\tbegin training'
    recs = train_model(model.train_func,model.predict,
                       train_x[:size],train_y[:size],
                       valid_x,valid_y,
                       lr0,lrdecay,bs,epochs)
    
    print '\tevaluating train/valid sets'
    evaluate_model(model.predict_proba,
                   train_x[:min(size,10000)],train_y[:min(size,10000)],
                   valid_x,valid_y)
    
    print '\tevaluating train/test sets'
    evaluate_model(model.predict_proba,
                   train_x[:min(size,10000)],train_y[:min(size,10000)],
                   test_x,test_y)



