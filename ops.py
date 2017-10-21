#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:54:50 2017

@author: Chin-Wei
"""


import cPickle as pickle
import gzip
from sklearn.preprocessing import OneHotEncoder
import numpy as np
floatX = 'float32'

def load_mnist(filename):
    try:
        tr,va,te = pickle.load(gzip.open('mnist.pkl.gz','r'))
    except:
        tr,va,te = pickle.load(gzip.open(filename,'r'))
    tr_x,tr_y = tr
    va_x,va_y = va
    te_x,te_y = te
    # doesn't work on hades :/ 
    enc = OneHotEncoder(10)
    tr_y = enc.fit_transform(tr_y.reshape((-1,1))).toarray().reshape(50000,10).astype(int)
    va_y = enc.fit_transform(va_y.reshape((-1,1))).toarray().reshape(10000,10).astype(int)    
    te_y = enc.fit_transform(te_y.reshape((-1,1))).toarray().reshape(10000,10).astype(int)
    f = lambda d:d.astype(floatX) 
    return (f(d) for d in [tr_x, tr_y, va_x, va_y, te_x, te_y])
    

def load_cifar10(filename,val=0.1,seed=1000):
    tr_x, tr_y, te_x, te_y = pickle.load(open(filename,'r'))
    enc = OneHotEncoder(10)
    tr_y = enc.fit_transform(tr_y).toarray().reshape(50000,10).astype(int)
    te_y = enc.fit_transform(te_y).toarray().reshape(10000,10).astype(int)
    
    n = tr_x.shape[0]
    trn_ind = set(range(n))
    rng = np.random.RandomState(seed)
    val_ind = rng.choice(n,int(n*val),False)
    trn_ind = np.array(list(trn_ind.difference(val_ind)))
    tr_x, tr_y, va_x, va_y = tr_x[trn_ind], tr_y[trn_ind], \
                             tr_x[val_ind], tr_y[val_ind]
                             
    f = lambda d:d.astype(floatX) 
    return (f(d) for d in [tr_x, tr_y, va_x, va_y, te_x, te_y])


def get_index(vec,key):
    return np.arange(vec.shape[0])[key(vec)]

def load_cifar5(filename,val=0.1,seed=1000):
    tr_x, tr_y, te_x, te_y = pickle.load(open(filename,'r'))
    tr_ind  = get_index(tr_y.flatten(),lambda y:y<=4)
    te_ind  = get_index(te_y.flatten(),lambda y:y<=4)
    tr_x, tr_y = tr_x[tr_ind], tr_y[tr_ind]
    te_x, te_y = te_x[te_ind], te_y[te_ind]
    enc = OneHotEncoder(5)
    tr_n, te_n = tr_x.shape[0], te_x.shape[0]
    tr_y = enc.fit_transform(tr_y).toarray().reshape(tr_n,5).astype(int)
    te_y = enc.fit_transform(te_y).toarray().reshape(te_n,5).astype(int)
    
    n = tr_x.shape[0]
    trn_ind = set(range(n))
    rng = np.random.RandomState(seed)
    val_ind = rng.choice(n,int(n*val),False)
    trn_ind = np.array(list(trn_ind.difference(val_ind)))
    tr_x, tr_y, va_x, va_y = tr_x[trn_ind], tr_y[trn_ind], \
                             tr_x[val_ind], tr_y[val_ind]
                                 
    f = lambda d:d.astype(floatX) 
    return (f(d) for d in [tr_x, tr_y, va_x, va_y, te_x, te_y])



