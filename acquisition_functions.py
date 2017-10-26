#!/usr/bin/env python
import os
import time

import numpy
np = numpy
from scipy.stats import mode

import theano
floatX = theano.config.floatX
import lasagne

# TODO: Dirichlet acquisition functions!

"""
All acquisition functions take inputs in the form:
    (num_samples, num_examples, num_outputs)
"""

from scipy.stats import entropy

# FIXME (get it from the cluster...)
def get_entropy(arr, axis=None):
    """ compute the entropy along a given axis """
    axis = axis % arr.ndim
    # NTS: only works if fn operates along 0-th axis!
    def along_axis(fn, arr, axis):
        """ perform fn along axis of arr """
        axes = range(arr.ndim)
        swap = [axes.pop(axis),] + axes
        return fn(arr.transpose(swap))
    return along_axis(entropy, arr, axis)

def bald(sampled_pys): 
    return get_entropy(sampled_pys.mean(axis=0), axis=-1) - np.mean(get_entropy(sampled_pys, axis=-1), axis=0)

def max_ent(sampled_pys):
    return get_entropy(sampled_pys.mean(axis=0), axis=-1)

def var_ratio(sampled_pys):
    return 1 - np.max(np.mean(sampled_pys, axis=0), axis=-1)

def mean_std(sampled_pys):
    return sampled_pys.std(0).mean(-1)

