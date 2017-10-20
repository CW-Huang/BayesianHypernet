#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)
from collections import OrderedDict
from time import time
import numpy as np
import pymc3 as pm
from scipy.misc import logsumexp
from scipy.stats import norm
import theano
import theano.tensor as T

# Utils we will be using elsewhere:


def unpack(v, weight_shapes):
    assert(np.ndim(v) == 1)

    D = OrderedDict()
    tt = 0
    for varname, ws in weight_shapes.iteritems():
        num_param = np.prod(ws, dtype=int)
        D[varname] = v[tt:tt + num_param].reshape(ws)
        tt += num_param
    # assert(tt == len(v))  # Check all used
    # Check order was preserved:
    assert(D.keys() == weight_shapes.keys())
    return D


def get_num_params(weight_shapes):
    cnt = sum(np.prod(ws, dtype=int) for ws in weight_shapes.itervalues())
    return cnt


def summarize(X, p=(0.25, 0.50, 0.75)):
    assert(X.ndim >= 2)
    p = np.asarray(p)
    assert(p.ndim == 1 and np.all(0.0 <= p) and np.all(p <= 1.0))
    LB_p = 0.5 * (1.0 - p)
    UB_p = 0.5 * (1.0 + p)

    LB = np.percentile(X, 100.0 * LB_p, axis=0)
    UB = np.percentile(X, 100.0 * UB_p, axis=0)
    return LB, UB


def mlp_pred(X, theta, n_layers, lib=T):
    '''Vanilla MLP ANN for regression. Can use lib=T or lin=np.'''
    yp = X
    for nn in xrange(n_layers):
        W, b = theta['W_' + str(nn)], theta['b_' + str(nn)]
        act = lib.dot(yp, W) + b[None, :]
        # Linear at final layer
        yp = act if nn == n_layers - 1 else lib.maximum(0.0, act)
    log_prec = theta['log_prec']
    assert(log_prec.ndim == 0)
    y_prec = lib.exp(log_prec)
    return yp, y_prec


def mlp_loglik_np(X, y, theta, n_layers):
    yp, y_prec = mlp_pred(X, theta, n_layers, lib=np)
    std_dev = np.sqrt(1.0 / y_prec)
    assert(std_dev.ndim == 0)
    assert(yp.shape == y.shape)

    loglik = norm.logpdf(y, loc=yp, scale=std_dev)
    assert(loglik.shape == y.shape)
    return np.sum(loglik)


def logprior_np(theta):
    logprior = sum(np.sum(norm.logpdf(th, loc=0.0, scale=1.0))
                   for th in theta.itervalues())
    return logprior

# Define theano only flat versions of MLP functions


def mlp_logprior_flat_tt(theta):
    '''Not vectorized, just single case.'''
    assert(theta.ndim == 1)
    # Standard Gauss
    logprior = -0.5 * T.sum(theta ** 2)  # Ignoring normalizing constant
    return logprior


def mlp_pred_flat_tt(X, theta, weight_shapes):
    theta_dict = unpack(theta, weight_shapes)
    n_layers, rem = divmod(len(theta_dict) - 1, 2)
    assert(rem == 0)

    yp, y_prec = mlp_pred(X, theta_dict, n_layers, lib=T)
    return yp, y_prec


def mlp_loglik_flat_tt(X, y, theta, weight_shapes):
    yp, y_prec = mlp_pred_flat_tt(X, theta, weight_shapes)
    err = yp - y

    loglik_c = -0.5 * np.log(2.0 * np.pi)
    loglik_cmp = 0.5 * T.log(y_prec)  # Hopefully Theano undoes log(exp())
    loglik_fit = -0.5 * y_prec * T.sum(err ** 2, axis=1)
    loglik = loglik_c + loglik_cmp + loglik_fit
    return loglik

# Now implement in pymc3


def make_normal_vars_pm(weight_shapes):
    D = OrderedDict()
    for varname, ws in weight_shapes.iteritems():
        D[varname] = pm.Normal(varname, 0.0, sd=1.0, shape=ws)
    # Check order was preserved:
    assert(D.keys() == weight_shapes.keys())
    return D


def primary_net_pm(X_train, Y_train, weight_shapes):
    '''Assume input data are shared variables.'''
    N, D = X_train.get_value().shape
    assert(Y_train.get_value().shape == (N, 1))
    # TODO check X_test
    n_layers, rem = divmod(len(weight_shapes) - 1, 2)
    assert(rem == 0)

    with pm.Model() as neural_network:
        # Build param_dict
        theta_dict = make_normal_vars_pm(weight_shapes)

        yp_train, y_prec = mlp_pred(X_train, theta_dict, n_layers, lib=T)

        assert(y_prec.ndim == 0)
        assert(yp_train.ndim == 2)
        assert(Y_train.ndim == 2)
        pm.Normal('out', mu=yp_train, tau=y_prec, observed=Y_train)

        #yp_test, _ = mlp_pred(X_test, theta_dict, n_layers, lib=T)
        #pm.Deterministic('yp_test', yp_test)
    return neural_network


def hmc_net(X_train, Y_train, X_test, Y_test, initializer_f, weight_shapes,
            restarts=100, n_iter=500, n_tune=10, init_scale_iter=1000):
    '''hypernet_f can serve as initializer_f. Y_test only used to monitor
    loglik'''
    num_params = get_num_params(weight_shapes)

    ann_input = theano.shared(X_train)
    ann_output = theano.shared(Y_train)
    ann_model = primary_net_pm(ann_input, ann_output, weight_shapes)

    theta0 = [initializer_f(np.random.randn(num_params), False)
              for _ in xrange(init_scale_iter)]
    assert(np.shape(theta0) == (init_scale_iter, num_params))
    var_estimate = np.var(theta0, axis=0)
    assert(var_estimate.shape == (num_params,))

    tr = [None] * restarts
    logprior = np.nan + np.zeros((n_tune + n_iter, restarts))
    loglik_train = np.nan + np.zeros((n_tune + n_iter, restarts))
    loglik_test = np.nan + np.zeros((n_tune + n_iter, restarts))
    logp_chk = np.nan + np.zeros((n_tune + n_iter, restarts))
    for rr in xrange(restarts):
        print 'HMC run', str(rr)

        z_noise = np.random.randn(num_params)
        theta_vec = initializer_f(z_noise, False)
        assert(theta_vec.shape == (num_params,))
        # Cast to ordinary dict to be safe, for now
        start = dict(unpack(theta_vec, weight_shapes))

        # TODO decide fairest way to deal with tuning period
        print 'starting to sample'
        t = time()
        with ann_model:
            step = pm.NUTS(scaling=var_estimate, is_cov=True)
            tr[rr] = pm.sampling.sample(draws=n_iter, step=step, start=start,
                                        progressbar=False, tune=n_tune,
                                        discard_tuned_samples=False)
        print (time() - t), 's'

        n_layers, rem = divmod(len(weight_shapes) - 1, 2)
        assert(rem == 0)
        for ii, theta in enumerate(tr[rr]):
            logp_chk[ii, rr] = ann_model.logp(theta)
            logprior[ii, rr] = logprior_np(theta)
            loglik_train[ii, rr] = mlp_loglik_np(X_train, Y_train, theta, n_layers)
            loglik_test[ii, rr] = mlp_loglik_np(X_test, Y_test, theta, n_layers)
    err = np.max(np.abs(logp_chk - (logprior + loglik_train)))
    print 'nrg log10 err %f' % np.log10(err)
    # Could use merge traces but prob not worth trouble
    return tr, (logprior, loglik_train, loglik_test)

# Post-process pymc3 traces


def hmc_pred(tr_list, X_test, n_layers, y_test=None, p=(0.025, 0.5, 0.975)):
    n_samples = len(tr_list)
    assert(n_samples >= 1)
    n_iter = len(tr_list[0])
    n_grid, _ = X_test.shape
    assert(y_test is None or y_test.shape == (n_grid,))

    mu_test = np.zeros((n_samples, n_iter, n_grid))
    y_samples = np.zeros((n_samples, n_iter, n_grid))
    loglik_raw = np.nan + np.zeros((n_samples, n_iter, n_grid))
    for ss, tr in enumerate(tr_list):
        assert(len(tr) == n_iter)

        noise = np.random.randn()
        for ii, theta in enumerate(tr):
            mu_test_, y_prec = mlp_pred(X_test, theta, n_layers, lib=np)
            mu_test[ss, ii, :] = mu_test_[:, 0]
            std_dev = np.sqrt(1.0 / y_prec)
            y_samples[ss, ii, :] = mu_test[ss, ii, :] + std_dev * noise
            if y_test is not None:
                loglik_raw[ss, ii, :] = \
                    norm.logpdf(y_test, loc=mu_test[ss, ii, :], scale=std_dev)

    # Get predictive loglik over iteration
    loglik = logsumexp(loglik_raw, axis=0) - np.log(n_samples)
    assert(loglik.shape == (n_iter, n_grid))
    loglik = np.mean(loglik, axis=1)  # Get average loss per example
    assert(loglik.shape == (n_iter,))

    # Summarize the predictions
    mu = np.mean(mu_test, axis=0)  # Average the means w/o extra noise
    std = np.std(mu_test, axis=0, ddof=0)  # MLE std
    LB, UB = summarize(y_samples, p)  # MC estimate quantiles using noise
    assert(mu.shape == (n_iter, n_grid))
    assert(std.shape == (n_iter, n_grid))
    assert(LB.shape == (len(p), n_iter, n_grid))
    assert(UB.shape == LB.shape)
    assert(np.all(LB <= UB))
    return mu, std, LB, UB, loglik, loglik_raw
