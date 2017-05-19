#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)

import lasagne
import theano

import numpy as np
from scipy.misc import logsumexp
import theano.tensor as T
import hypernet_trainer as ht
import ign.ign as ign
from ign.t_util import make_shared_dict, make_unshared_dict

NOISE_STD = 0.02


def dm_example(N):
    x = 0.5 * np.random.rand(N, 1)

    noise = NOISE_STD * np.random.randn(N, 1)
    x_n = x + noise
    y = x + 0.3 * np.sin(2.0 * np.pi * x_n) + 0.3 * np.sin(4.0 * np.pi * x_n) \
        + noise
    return x, y


def unpack(v, weight_shapes):
    L = []
    tt = 0
    for ws in weight_shapes:
        num_param = np.prod(ws, dtype=int)
        L.append(v[tt:tt + num_param].reshape(ws))
        tt += num_param
    return L


def primary_net_f(X, theta, weight_shapes):
    L = unpack(theta, weight_shapes)
    n_layers, rem = divmod(len(L) - 1, 2)
    assert(rem == 0)

    yp = X
    for nn in xrange(n_layers):
        W, b = L[2 * nn], L[2 * nn + 1]
        assert(W.ndim == 2 and b.ndim == 1)
        act = T.dot(yp, W) + b[None, :]
        # Linear at final layer
        yp = act if nn == n_layers - 1 else T.maximum(0.0, act)

    log_prec = L[-1]
    assert(log_prec.ndim == 0)
    y_prec = T.exp(log_prec)
    return yp, y_prec


def loglik_primary_f(X, y, theta, weight_shapes):
    yp, y_prec = primary_net_f(X, theta, weight_shapes)
    err = yp - y

    loglik_c = -0.5 * np.log(2.0 * np.pi)
    loglik_cmp = 0.5 * T.log(y_prec)  # Hopefully Theano undoes log(exp())
    loglik_fit = -0.5 * y_prec * T.sum(err ** 2, axis=1)
    loglik = loglik_c + loglik_cmp + loglik_fit
    return loglik


def logprior_f(theta):
    # Standard Gauss
    logprior = -0.5 * T.sum(theta ** 2)  # Ignoring normalizing constant
    return logprior


def simple_test(X, y, X_valid, y_valid,
                n_epochs, n_batch, init_lr, weight_shapes,
                n_layers=5, vis_freq=100, n_samples=100,
                z_std=1.0):
    '''z_std = 1.0 gives the correct answer but other values might be good for
    debugging and analysis purposes.'''
    N, D = X.shape
    N_valid = X_valid.shape[0]
    assert(y.shape == (N, 1))  # Univariate for now
    assert(X_valid.shape == (N_valid, D) and y_valid.shape == (N_valid, 1))

    num_params = sum(np.prod(ws, dtype=int) for ws in weight_shapes)

    WL_init = 5 * 1e-2  # 10.0 ** (-2.0 / n_layers)
    layers = ign.init_ign_LU(n_layers, num_params, WL_val=WL_init)
    phi_shared = make_shared_dict(layers, '%d%s')

    ll_primary_f = lambda X, y, w: loglik_primary_f(X, y, w, weight_shapes)
    hypernet_f = lambda z, prelim: ign.network_T_and_J_LU(z[None, :], phi_shared, force_diag=prelim)[0][0, :]
    # TODO verify this length of size 1
    log_det_dtheta_dz_f = lambda z, prelim: T.sum(ign.network_T_and_J_LU(z[None, :], phi_shared, force_diag=prelim)[1])
    primary_f = lambda X, w: primary_net_f(X, w, weight_shapes)
    params_to_opt = phi_shared.values()
    R = ht.build_trainer(params_to_opt, N, ll_primary_f, logprior_f,
                         hypernet_f, primary_f=primary_f,
                         log_det_dtheta_dz_f=log_det_dtheta_dz_f)
    trainer, get_err, test_loglik, primary_out, grad_f = R

    batch_order = np.arange(int(N / n_batch))

    cost_hist = np.zeros(n_epochs)
    loglik_valid = np.zeros(n_epochs)
    for epoch in xrange(n_epochs):
        np.random.shuffle(batch_order)

        cost = 0.0
        for ii in batch_order:
            x_batch = X[ii * n_batch:(ii + 1) * n_batch]
            y_batch = y[ii * n_batch:(ii + 1) * n_batch]
            z_noise = z_std * np.random.randn(num_params)
            if epoch <= 1000:
                current_lr = init_lr
                prelim = False
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
    return phi, cost_hist, loglik_valid, primary_out, grad_f


def traditional_test(X, y, X_valid, y_valid, n_epochs, n_batch, init_lr, weight_shapes):
    '''z_std = 1.0 gives the correct answer but other values might be good for
    debugging and analysis purposes.'''
    N, D = X.shape
    N_valid = X_valid.shape[0]
    assert(y.shape == (N, 1))  # Univariate for now
    assert(X_valid.shape == (N_valid, D) and y_valid.shape == (N_valid, 1))

    num_params = sum(np.prod(ws, dtype=int) for ws in weight_shapes)
    phi_shared = make_shared_dict({'w': np.random.randn(num_params)})

    X_ = T.matrix('x')
    y_ = T.matrix('y')  # Assuming multivariate output
    lr = T.scalar('lr')

    loglik = loglik_primary_f(X_, y_, phi_shared['w'], weight_shapes)
    loss = -T.sum(loglik)
    params_to_opt = phi_shared.values()
    grads = T.grad(loss, params_to_opt)
    updates = lasagne.updates.adam(grads, params_to_opt, learning_rate=lr)

    test_loglik = theano.function([X_, y_], loglik)
    trainer = theano.function([X_, y_, lr], loss, updates=updates)
    primary_out = theano.function([X_], primary_net_f(X_, phi_shared['w'], weight_shapes))

    batch_order = np.arange(int(N / n_batch))

    cost_hist = np.zeros(n_epochs)
    loglik_valid = np.zeros(n_epochs)
    for epoch in xrange(n_epochs):
        np.random.shuffle(batch_order)

        cost = 0.0
        for ii in batch_order:
            x_batch = X[ii * n_batch:(ii + 1) * n_batch]
            y_batch = y[ii * n_batch:(ii + 1) * n_batch]
            current_lr = init_lr
            batch_cost = trainer(x_batch, y_batch, current_lr)
            cost += batch_cost
        cost /= len(batch_order)
        print cost
        cost_hist[epoch] = cost
        loglik_valid[epoch] = np.mean(test_loglik(X, y))
        print 'valid %f' % loglik_valid[epoch]

    phi = make_unshared_dict(phi_shared)
    return phi, cost_hist, loglik_valid, primary_out


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(5645)

    init_lr = 0.0005
    n_epochs = 500
    n_batch = 32
    N = 500
    z_std = 1.0  # 1.0 is correct for the model, 0.0 is MAP

    primary_layers = 1
    input_dim = 1
    hidden_dim = 50
    output_dim = 1

    weight_shapes = [(input_dim, hidden_dim), (hidden_dim,),
                     (hidden_dim, output_dim), (output_dim,), ()]

    X, y = dm_example(N)
    X_valid, y_valid = dm_example(N)

    phi_trad, cost_hist_trad, loglik_valid_trad, primary_out_trad = \
        traditional_test(X, y, X_valid, y_valid, n_epochs, n_batch, init_lr, weight_shapes)

    phi, cost_hist, loglik_valid, primary_out, grad_f = \
        simple_test(X, y, X_valid, y_valid,
                    n_epochs, n_batch, init_lr, weight_shapes,
                    n_layers=3, z_std=z_std)

    n_samples = 500
    n_grid = 1000
    num_params = sum(np.prod(ws, dtype=int) for ws in weight_shapes)

    x_grid = np.linspace(-0.5, 1.0, n_grid)
    mu_trad, prec_trad = primary_out_trad(x_grid[:, None])
    std_dev_trad = np.sqrt(1.0 / prec_trad)

    mu_grid = np.zeros((n_samples, n_grid))
    y_grid = np.zeros((n_samples, n_grid))
    for ss in xrange(n_samples):
        z_noise = z_std * np.random.randn(num_params)
        mu, prec = primary_out(x_grid[:, None], z_noise, False)
        std_dev = np.sqrt(1.0 / prec)
        # Just store an MC version than do the mixture of Gauss logic
        mu_grid[ss, :] = mu[:, 0]
        # Note: using same noise across whole grid
        y_grid[ss, :] = mu[:, 0] + std_dev * np.random.randn()

    _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.plot(X[:100,:], y[:100], 'rx', zorder=0)
    ax1.plot(x_grid, mu_grid[:5, :].T, zorder=1, alpha=0.7)
    ax1.plot(x_grid, np.mean(mu_grid, axis=0), 'k', zorder=2)
    ax1.plot(x_grid, np.percentile(y_grid, 2.5, axis=0), 'k--', zorder=2)
    ax1.plot(x_grid, np.percentile(y_grid, 97.5, axis=0), 'k--', zorder=2)
    ax1.grid()
    ax1.set_title('hypernet')

    ax2.plot(X[:100,:], y[:100], 'rx', zorder=0)
    ax2.plot(x_grid, mu_trad, 'k', zorder=2)
    ax2.plot(x_grid, mu_trad - 2 * std_dev_trad, 'k--', zorder=2)
    ax2.plot(x_grid, mu_trad + 2 * std_dev_trad, 'k--', zorder=2)
    ax2.grid()
    ax2.set_title('traditional')

    plt.figure()
    plt.plot(loglik_valid, label='hypernet')
    plt.plot(loglik_valid_trad, label='traditional')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('validation log likelihood')
    plt.grid()

    plt.figure()
    plt.plot(cost_hist, label='hypernet')
    plt.xlabel('epoch')
    plt.ylabel('training cost (-ELBO)')
    plt.grid()
