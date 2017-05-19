#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)
from collections import OrderedDict
import numpy as np
import scipy.special as sc
import theano
import theano.tensor as T


def get_shape(X):
    if isinstance(X, theano.tensor.sharedvar.TensorSharedVariable):
        shape = X.get_value().shape
    else:
        shape = np.shape(X)
    # Passed theano tensor or something weird has gone wrong
    assert(type(shape) == tuple)
    return shape


def make_shared(v, name, bc=None):
    if bc is None:
        sv = theano.shared(v, name=name)
    else:
        sv = theano.shared(v, name=name, broadcastable=bc)
    return sv.astype(theano.config.floatX)


def make_shared_dict(param_dict_np, name_pat='%s', bc={}):
    param_dict_shared = \
        OrderedDict((k, make_shared(v, name_pat % k, bc.get(k, None)))
                    for k, v in param_dict_np.iteritems())
    return param_dict_shared


def make_unshared_dict(param_dict_shared):
    param_dict_np = OrderedDict((k, v.get_value())
                                for k, v in param_dict_shared.iteritems())
    return param_dict_np


def log_abs_det_T(W):
    # TODO option for stable way
    return T.log(T.abs_(T.nlinalg.det(W)))


def log_abs_det_tri_T(L):
    '''Warning! only valid for triangular matrices.'''
    return T.sum(T.log(T.abs_(T.nlinalg.extract_diag(L))))


def L2_T(T_var):
    return T.sum(T_var ** 2)


def norm_logpdf_T(x):
    Z_np = -0.5 * np.log(2.0 * np.pi)
    logpdf = Z_np - 0.5 * (x ** 2)
    return logpdf


def t_logpdf_T(x, df=2.0):
    # TODO do test here
    # TODO reconsider default here
    # Assuming df is not itself Theano variable for now
    df = np.asarray(df, dtype=float)
    assert(df.ndim == 0)

    Z_np = sc.gammaln((df + 1.0) / 2.0) - \
        (0.5 * np.log(df * np.pi) + sc.gammaln(df / 2.0))
    fac_np = (df + 1.0) / 2.0
    logpdf = Z_np - fac_np * np.log(1.0 + (x ** 2) / df)
    return logpdf


def logistic_logpdf_T(x, c=1.0):
    '''Theano version of ss.logistic.logpdf().'''
    # TODO do test here
    logpdf = T.log(c) - x - (c + 1.0) * T.log1p(T.exp(-x))
    return logpdf


def get_adam_min_updates(param_list, gradients, epoch, lr_decay=1.0,
                         b1=0.95, b2=0.999, base_learning_rate=0.001):
    # Also similar to:
    # https://gist.github.com/Newmu/acb738767acb4788bac3
    epsilon = 1e-8  # Something small
    learning_rate = base_learning_rate / lr_decay

    # Setup
    m_dict = OrderedDict()
    v_dict = OrderedDict()
    for parameter in param_list:
        name = parameter.name
        curr_shape = parameter.get_value().shape
        m_dict[name] = make_shared(np.zeros(curr_shape), name='m_' + name)
        v_dict[name] = make_shared(np.zeros(curr_shape), name='v_' + name)
    zipped_up = zip(param_list, gradients, m_dict.values(), v_dict.values())

    # Theano part
    gamma = T.sqrt(1.0 - b2 ** epoch) / (1.0 - b1 ** epoch)

    updates = OrderedDict()
    for parameter, gradient, m, v in zipped_up:
        new_m = b1 * m + (1.0 - b1) * gradient
        new_v = b2 * v + (1.0 - b2) * (gradient ** 2)

        delta = learning_rate * gamma * new_m / (T.sqrt(new_v) + epsilon)
        updates[parameter] = parameter - delta

        updates[m] = new_m
        updates[v] = new_v
    return updates
