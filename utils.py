# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:39:54 2017

@author: Chin-Wei
"""

import theano.tensor as T
import numpy as np

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
    return c - log_var/2. - (x - mean)**2 / (2. * T.exp(log_var) + eps)


def log_laplace(x,mean,inv_scale,epsilon=1e-7):
    return - T.log(2*(inv_scale+epsilon)) - T.abs_(x-mean)/(inv_scale+epsilon)



def softmax(x,axis=1):
    x_max = T.max(x, axis=axis, keepdims=True)
    exp = T.exp(x-x_max)
    return exp / T.sum(exp, axis=axis, keepdims=True)
    
    
    