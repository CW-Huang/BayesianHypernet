#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)

# https://stats.stackexchange.com/questions/113851/bayesian-estimation-of-n-of-a-binomial-distribution

import numpy as np
from scipy.misc import logsumexp
from scipy.special import expit as logistic
import theano
import theano.tensor as T
import hypernet_trainer as ht
import ign.ign as ign
from ign.t_util import make_shared_dict, make_unshared_dict

theano.config.on_unused_input = 'warn'


def loglik_primary_f(k, y, theta, lower_n):
    logit_p = theta[0]
    logn = theta[1]
    n = lower_n + T.exp(logn)
    k = k[:, 0]

    p = T.nnet.nnet.sigmoid(logit_p)

    combiln = T.gammaln(n + 1) - (T.gammaln(k + 1) + T.gammaln(n - k + 1))
    # add y to stop theano from complaining
    #loglik = combiln + k * T.log(p) + (n - k) * T.log1p(-p) + 0.0 * T.sum(y)
    loglik = combiln + k * T.log(p) + (n - k) * T.log(1.0 - p) + 0.0 * T.sum(y)
    return loglik


def logprior_f(theta, lower_n):
    logit_p = theta[0]
    logn = theta[1]

    logprior_n = logn - T.log(lower_n + T.exp(logn))
    logprior_p = T.log(T.nnet.nnet.sigmoid(logit_p)) + T.log(T.nnet.nnet.sigmoid(-logit_p))

    logprior = logprior_n + logprior_p
    return logprior


def simple_test(X, X_valid, n_epochs, n_batch, init_lr,
                n_layers=5, vis_freq=100, n_samples=100,
                z_std=1.0):
    '''z_std = 1.0 gives the correct answer but other values might be good for
    debugging and analysis purposes.'''
    N, D = X.shape
    N_valid = X_valid.shape[0]
    assert(X_valid.shape == (N_valid, D))
    y_valid = np.zeros_like(X_valid)

    num_params = 2
    max_x = np.max(X)

    WL_init = 1.0
    layers = ign.init_ign_LU(n_layers, num_params, WL_val=WL_init)
    phi_shared = make_shared_dict(layers, '%d%s')

    ll_primary_f = lambda X, y, w: loglik_primary_f(X, y, w, max_x)
    logprior_f_corrected = lambda theta: logprior_f(theta, max_x)
    hypernet_f = lambda z, prelim: ign.network_T_and_J_LU(z[None, :], phi_shared, force_diag=prelim)[0][0, :]
    log_det_dtheta_dz_f = lambda z, prelim: T.sum(ign.network_T_and_J_LU(z[None, :], phi_shared, force_diag=prelim)[1])
    params_to_opt = phi_shared.values()
    R = ht.build_trainer(params_to_opt, N, ll_primary_f, logprior_f_corrected,
                         hypernet_f, log_det_dtheta_dz_f=log_det_dtheta_dz_f)
    trainer, get_err, test_loglik, _, grad_f = R

    batch_order = np.arange(int(N / n_batch))

    cost_hist = np.zeros(n_epochs)
    loglik_valid = np.zeros(n_epochs)
    for epoch in xrange(n_epochs):
        np.random.shuffle(batch_order)

        cost = 0.0
        for ii in batch_order:
            x_batch = X[ii * n_batch:(ii + 1) * n_batch]
            y_batch = np.zeros_like(x_batch)
            z_noise = z_std * np.random.randn(num_params)
            if epoch <= 200:
                current_lr = init_lr
                prelim = True
            else:
                current_lr = init_lr * 0.01
                prelim = False
            batch_cost = trainer(x_batch, y_batch, z_noise, current_lr, prelim)
            cost += batch_cost
        cost /= len(batch_order)
        print cost
        cost_hist[epoch] = cost

        loglik_valid_s = np.zeros((N_valid, n_samples))
        for ss in xrange(n_samples):
            z_noise = z_std * np.random.randn(num_params)
            loglik_valid_s[:, ss] = test_loglik(X_valid, y_valid, z_noise, False)
        loglik_valid_s_adj = loglik_valid_s - np.log(n_samples)
        loglik_valid[epoch] = np.mean(logsumexp(loglik_valid_s_adj, axis=1))
        print 'valid %f' % loglik_valid[epoch]

    phi = make_unshared_dict(phi_shared)
    return phi, cost_hist, loglik_valid, grad_f


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(5645)

    init_lr = 0.001
    n_epochs = 100
    n_batch = 5
    N = 1000
    z_std = 1.0  # 1.0 is correct for the model, 0.0 is MAP

    X = np.array([[53, 57, 66, 67, 72]]).T
    #X = np.random.binomial(100, 0.7, 500)
    #X = X[:, None]

    R = simple_test(X, X, n_epochs, n_batch, init_lr, n_layers=5, z_std=z_std)
    phi, cost_hist, loglik_valid, grad_f = R

    phi_W = ign.LU_to_W_np(phi)
    theta = np.array([ign.ign_rnd(phi_W)[0][0, :] for _ in xrange(1000)])
    plt.plot(np.max(X) + np.exp(theta[:, 1]), logistic(theta[:, 0]), '.')
    plt.xlim([np.max(X), 2000])
    plt.ylim([0, 1])
