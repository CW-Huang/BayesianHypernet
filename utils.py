# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:39:54 2017

@author: Chin-Wei
"""

import theano.tensor as T
import numpy as np

from lasagne.init import Normal
from lasagne.init import Initializer, Orthogonal

c = - 0.5 * T.log(2*np.pi)

def log_sum_exp(A, axis=None, sum_op=T.sum):

    A_max = T.max(A, axis=axis, keepdims=True)
    B = T.log(sum_op(T.exp(A - A_max), axis=axis, keepdims=True)) + A_max

    if axis is None:
        return B.dimshuffle(())  # collapse to scalar
    else:
        if not hasattr(axis, '__iter__'): axis = [axis]
        return B.dimshuffle([d for d in range(B.ndim) if d not in axis])  
        # drop summed axes

def log_mean_exp(A, axis=None,weights=None):
    if weights:
        return log_sum_exp(A, axis, sum_op=weighted_sum(weights))
    else:
        return log_sum_exp(A, axis, sum_op=T.mean)


def weighted_sum(weights):
    return lambda A,axis,keepdims: T.sum(A*weights,axis=axis,keepdims=keepdims)    


def log_stdnormal(x):
    return c - 0.5 * x**2 

def log_normal(x,mean,log_var,eps=0.0):
    if type(x) == list:
        x = T.concatenate([w.flatten() for w in x])
    return c - log_var/2. - (x - mean)**2 / (2. * T.exp(log_var) + eps)


def log_laplace(x,mean,inv_scale,epsilon=1e-7):
    return - T.log(2*(inv_scale+epsilon)) - T.abs_(x-mean)/(inv_scale+epsilon)


def log_scale_mixture_normal(x,m,log_var1,log_var2,p1,p2):
    axis = x.ndim
    log_n1 = T.log(p1)+log_normal(x,m,log_var1)
    log_n2 = T.log(p2)+log_normal(x,m,log_var2)
    log_n_ = T.stack([log_n1,log_n2],axis=axis)
    log_n = log_sum_exp(log_n_,-1)
    return log_n.sum(-1)


def softmax(x,axis=1):
    x_max = T.max(x, axis=axis, keepdims=True)
    exp = T.exp(x-x_max)
    return exp / T.sum(exp, axis=axis, keepdims=True)



# inds : the indices of the examples you wish to evaluate
#   these should probably be ALL of the inds, OR be randomly sampled
def MCpred(X, predict_probs_fn=None, num_samples=100, inds=None, returns='preds', num_classes=10):
    if inds is None:
        inds = range(len(X))
    rval = np.empty((num_samples, len(inds), num_classes))
    for ind in range(num_samples):
        rval[ind] = predict_probs_fn(X[inds])
    if returns == 'samples':
        return rval
    elif returns == 'probs':
        return rval.mean(0)
    elif returns == 'preds':
        return rval.mean(0).argmax(-1)


    
    
# TODO
class DanNormal(Initializer):
    def __init__(self, initializer=Normal, nonlinearity='relu', c01b=False, dropout_p=0.):
        if nonlinearity == 'relu':
            g1 = g2 = .5
        elif nonlinearity == 'gelu':
            g1 = .425
            g2 = .444

        p = 1 - dropout_p
        self.denominator = (g1 / p + p * g2)**.5

        self.__dict__.update(locals())

    def sample(self, shape):
        if self.c01b:
            assert False
            if len(shape) != 4:
                raise RuntimeError(
                    "If c01b is True, only shapes of length 4 are accepted")

            n1, n2 = shape[0], shape[3]
            receptive_field_size = shape[1] * shape[2]
        else:
            if len(shape) < 2:
                raise RuntimeError(
                    "This initializer only works with shapes of length >= 2")

            n1, n2 = shape[:2]
            receptive_field_size = np.prod(shape[2:])

        std = self.gain * np.sqrt(2.0 / ((n1 + n2) * receptive_field_size))
        # TODO: orthogonal
        return self.initializer(std=std).sample(shape)


def train_model(model,X,Y,Xv,Yv,
                lr0=0.001,lrdecay=1,bs=20,epochs=50,anneal=0,name='0',
                e0=0,rec=0,print_every=100,v_mc=20,n_classes=10):
    
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
    va_rec_name = name+'_recs'
    save_path = name + '.params'
    va_recs = list()
    tr_recs = list()
    
    t = 0
    for e in range(epochs):
        
        if e <= e0:
            continue
        
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
            
            loss = model.train_func(x,y,N,lr,w)
            
            if t%print_every==0:
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)
                tr_acc = (model.predict(X)==Y.argmax(1)).mean()
                va_acc = (model.predict(Xv)==Yv.argmax(1)).mean()
                print '\ttrain acc: {}'.format(tr_acc)
                print '\tvalid acc: {}'.format(va_acc)
            t+=1
        
        va_acc = evaluate_model(model.predict_proba,Xv,Yv,n_mc=v_mc,
                                n_classes=n_classes)
        print '\n\nva acc at epochs {}: {}'.format(e,va_acc)    
        
        va_recs.append(va_acc)
        
        if va_acc > rec:
            print '.... save best model .... '
            model.save(save_path,[e])
            rec = va_acc
    
            with open(va_rec_name,'a') as rec_file:
                for r in va_recs:
                    rec_file.write(str(r)+'\n')
            
            va_recs = list()
            
        print '\n\n'
    



def evaluate_model(predict_proba,X,Y,n_mc=100,max_n=100,n_classes=10):
    MCt = np.zeros((n_mc,X.shape[0],n_classes))
    
    N = X.shape[0]
    num_batches = np.ceil(N / float(max_n)).astype(int)
    for i in range(n_mc):
        for j in range(num_batches):
            x = X[j*max_n:(j+1)*max_n]
            MCt[i,j*max_n:(j+1)*max_n] = predict_proba(x)
    
    Y_pred = MCt.mean(0).argmax(-1)
    Y_true = Y.argmax(-1)
    return np.equal(Y_pred,Y_true).mean()