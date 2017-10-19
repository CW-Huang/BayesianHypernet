#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)

from collections import OrderedDict
import cPickle as pkl
import lasagne
import theano

import numpy as np
from scipy.misc import logsumexp
import theano.tensor as T
import hypernet_trainer as ht
import ign.ign as ign
from ign.t_util import make_shared_dict, make_unshared_dict
import mlp_hmc

NOISE_STD = 0.02


def dm_example(N):
    x = 0.5 * np.random.rand(N, 1)

    noise = NOISE_STD * np.random.randn(N, 1)
    x_n = x + noise
    y = x + 0.3 * np.sin(2.0 * np.pi * x_n) + 0.3 * np.sin(4.0 * np.pi * x_n) \
        + noise
    return x, y


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

    num_params = mlp_hmc.get_num_params(weight_shapes)

    WL_init = 1e-2  # 10.0 ** (-2.0 / n_layers)
    layers0 = ign.init_ign_LU(n_layers, num_params, WL_val=WL_init)
    phi_shared = make_shared_dict(layers0, '%d%s')

    # TODO use innfer functions to make this simpler

    ll_primary_f = lambda X, y, w: mlp_hmc.mlp_loglik_flat_tt(X, y, w, weight_shapes)
    hypernet_f = lambda z, prelim=False: ign.network_T_and_J_LU(z[None, :], phi_shared, force_diag=prelim)[0][0, :]
    # TODO verify this length of size 1
    log_det_dtheta_dz_f = lambda z, prelim: T.sum(ign.network_T_and_J_LU(z[None, :], phi_shared, force_diag=prelim)[1])
    primary_f = lambda X, w: mlp_hmc.mlp_pred_flat_tt(X, w, weight_shapes)
    params_to_opt = phi_shared.values()
    R = ht.build_trainer(params_to_opt, N, ll_primary_f, mlp_hmc.mlp_logprior_flat_tt,
                         hypernet_f, primary_f=primary_f,
                         log_det_dtheta_dz_f=log_det_dtheta_dz_f)
    trainer, get_err, test_loglik, primary_out, grad_f, theta_f = R

    batch_order = np.arange(int(N / n_batch))

    burn_in = 100

    cost_hist = np.zeros(n_epochs)
    loglik_valid = np.zeros(n_epochs)
    for epoch in xrange(n_epochs):
        np.random.shuffle(batch_order)

        cost = 0.0
        for ii in batch_order:
            x_batch = X[ii * n_batch:(ii + 1) * n_batch]
            y_batch = y[ii * n_batch:(ii + 1) * n_batch]
            z_noise = z_std * np.random.randn(num_params)
            if epoch <= burn_in:
                current_lr = init_lr
                prelim = False
            else:
                decay_fac = 1.0 - (float(epoch - burn_in) / (n_epochs - burn_in))
                current_lr = init_lr * decay_fac
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
    return phi, cost_hist, loglik_valid, primary_out, grad_f, theta_f


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

    loglik = mlp_hmc.mlp_loglik_flat_tt(X_, y_, phi_shared['w'], weight_shapes)
    # TODO prior in here??
    loss = -T.sum(loglik)
    params_to_opt = phi_shared.values()
    grads = T.grad(loss, params_to_opt)
    updates = lasagne.updates.adam(grads, params_to_opt, learning_rate=lr)

    test_loglik = theano.function([X_, y_], loglik)
    trainer = theano.function([X_, y_, lr], loss, updates=updates)
    primary_out = theano.function([X_], mlp_hmc.mlp_pred_flat_tt(X_, phi_shared['w'], weight_shapes))

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
    n_epochs = 1000
    n_batch = 32
    N = 1000
    z_std = 1.0  # 1.0 is correct for the model, 0.0 is MAP

    primary_layers = 1
    input_dim = 1
    hidden_dim = 50
    output_dim = 1

    weight_shapes = OrderedDict([('W_0', (input_dim, hidden_dim)),
                                 ('b_0', (hidden_dim,)),
                                 ('W_1', (hidden_dim, output_dim)),
                                 ('b_1', (output_dim,)),
                                 ('log_prec', ())])

    X, y = dm_example(N)
    X_valid, y_valid = dm_example(N)

    n_samples = 500
    n_grid = 1000
    x_grid = np.linspace(-0.5, 1.5, n_grid)

    phi, cost_hist, loglik_valid, primary_out, grad_f, hypernet_f = \
        simple_test(X, y, X_valid, y_valid,
                    n_epochs, n_batch, init_lr, weight_shapes,
                    n_layers=3, z_std=z_std)

    phi_trad, cost_hist_trad, loglik_valid_trad, primary_out_trad = \
        traditional_test(X, y, X_valid, y_valid, n_epochs, n_batch, init_lr, weight_shapes)

    tr = mlp_hmc.hmc_net(X, y, x_grid[:, None], hypernet_f, weight_shapes, restarts=50, n_iter=20)
    mu_hmc, LB_hmc, UB_hmc = mlp_hmc.hmc_pred(tr, x_grid[:, None], n_layers=primary_layers, chk=True)

    num_params = sum(np.prod(ws, dtype=int) for ws in weight_shapes)

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

    with open('reg_example_dump.pkl', 'wb') as f:
        pkl.dump((x_grid, mu_hmc, LB_hmc, UB_hmc, mu_grid, y_grid, mu_trad, std_dev_trad), f)

    _, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.plot(X[:100, :], y[:100], 'rx', zorder=0)
    ax1.plot(x_grid, mu_grid[:5, :].T, zorder=1, alpha=0.7)
    ax1.plot(x_grid, np.mean(mu_grid, axis=0), 'k', zorder=2)
    ax1.plot(x_grid, np.percentile(y_grid, 2.5, axis=0), 'k--', zorder=2)
    ax1.plot(x_grid, np.percentile(y_grid, 97.5, axis=0), 'k--', zorder=2)
    ax1.grid()
    ax1.set_title('hypernet')

    ax2.plot(X[:100, :], y[:100], 'rx', zorder=0)
    ax2.plot(x_grid, mu_trad, 'k', zorder=2)
    ax2.plot(x_grid, mu_trad - 2 * std_dev_trad, 'k--', zorder=2)
    ax2.plot(x_grid, mu_trad + 2 * std_dev_trad, 'k--', zorder=2)
    ax2.grid()
    ax2.set_title('traditional')
    plt.xlim([-0.2, 1.3])
    plt.ylim([-0.5, 1.2])

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
