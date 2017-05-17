#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 22:54:50 2017

@author: Chin-Wei
"""


import cPickle as pickle
import gzip
from sklearn.preprocessing import OneHotEncoder
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
    

def load_cifar10(filename):
    tr_x, tr_y, te_x, te_y = pickle.load(open(filename,'r'))
    enc = OneHotEncoder(10)
    tr_y = enc.fit_transform(tr_y).toarray().reshape(50000,10).astype(int)
    te_y = enc.fit_transform(te_y).toarray().reshape(10000,10).astype(int)
    f = lambda d:d.astype(floatX) 
    return (f(d) for d in [tr_x, tr_y, te_x, te_y])
