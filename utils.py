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
