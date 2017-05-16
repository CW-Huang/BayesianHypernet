# Ryan Turner (turnerry@iro.umontreal.ca)
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
    W0, b0, W1, b1, log_prec = unpack(theta, weight_shapes)

    a1 = T.maximum(0.0, T.dot(X, W0) + b0[None, :])
    yp = T.dot(a1, W1) + b1[None, :]

    y_prec = T.exp(log_prec)
    #y_prec = T.constant(1.0)
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
    #logprior = T.constant(0.0)  # TODO undo
    return logprior


def simple_test(X, y, X_valid, y_valid,
                n_epochs, n_batch, init_lr, weight_shapes,
                n_layers=5, vis_freq=100, n_samples=100,
                z_std=1.0):
    N, D = X.shape
    N_valid = X_valid.shape[0]
    assert(y.shape == (N, 1))  # Univariate for now
    assert(X_valid.shape == (N_valid, D) and y_valid.shape == (N_valid, 1))

    num_params = sum(np.prod(ws, dtype=int) for ws in weight_shapes)

    #layers = ign.init_ign(n_layers, num_params, rnd_W=False)

    layers = {}
    #layers[(0, 'WL')] = 1e-2 * np.ones(num_params)
    layers[(0, 'WL')] = 1e-2 * np.ones(num_params)
    layers[(0, 'bL')] = np.random.randn(num_params)
    # layers[(0, 'bL')][-1] = 6.0  # Start with big precision
    phi_shared_real = make_shared_dict(layers, '%d%s')
    phi_shared = phi_shared_real.copy()
    phi_shared[(0, 'WL')] = T.nlinalg.alloc_diag(phi_shared_real[(0, 'WL')])

    ll_primary_f = lambda X, y, w: loglik_primary_f(X, y, w, weight_shapes)
    hypernet_f = lambda z: ign.network_T_and_J(z[None, :], phi_shared)[0][0, :]
    # TODO verify this length of size 1
    log_det_dtheta_dz_f = lambda z: T.sum(ign.network_T_and_J(z[None, :], phi_shared)[1])
    primary_f = lambda X, w: primary_net_f(X, w, weight_shapes)

    params_to_opt = [phi_shared_real[(0, 'bL')]]  # TODO remove
    R = ht.build_trainer(params_to_opt, N, ll_primary_f, logprior_f,
                         hypernet_f, primary_f=primary_f,
                         log_det_dtheta_dz_f=log_det_dtheta_dz_f)
    trainer_prelim = R[0]

    params_to_opt = phi_shared_real.values()
    R = ht.build_trainer(params_to_opt, N, ll_primary_f, logprior_f,
                         hypernet_f, primary_f=primary_f,
                         log_det_dtheta_dz_f=log_det_dtheta_dz_f)
    trainer, get_err, test_loglik, primary_out = R

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
            if epoch <= -1:
                current_lr = init_lr
                batch_cost = trainer_prelim(x_batch, y_batch, z_noise, current_lr)
            else:
                current_lr = init_lr
                batch_cost = trainer(x_batch, y_batch, z_noise, current_lr)
            cost += batch_cost
        cost /= len(batch_order)
        print cost
        cost_hist[epoch] = cost

        loglik_valid_s = np.zeros((N_valid, n_samples))
        for ss in xrange(n_samples):
            z_noise = z_std * np.random.randn(num_params)
            loglik_valid_s[:, ss] = test_loglik(X_valid, y_valid, z_noise)
        loglik_valid_s_adj = loglik_valid_s - np.log(n_samples)
        loglik_valid[epoch] = np.mean(logsumexp(loglik_valid_s_adj, axis=1))
        print 'valid %f' % loglik_valid[epoch]

    phi = make_unshared_dict(phi_shared_real)
    return phi, cost_hist, loglik_valid, primary_out


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(5645)

    init_lr = 0.001
    n_epochs = 500
    n_batch = 32
    N = 1000
    z_std = 1.0  # 1.0 is correct for the model, 0.0 is MAP

    primary_layers = 1
    input_dim = 1
    hidden_dim = 50
    output_dim = 1

    weight_shapes = [(input_dim, hidden_dim), (hidden_dim,),
                     (hidden_dim, output_dim), (output_dim,), ()]

    X, y = dm_example(N)
    X_valid, y_valid = dm_example(N)
    phi, cost_hist, loglik_valid, primary_out = \
        simple_test(X, y, X_valid, y_valid,
                    n_epochs, n_batch, init_lr, weight_shapes,
                    n_layers=1, z_std=z_std)

    n_samples = 500
    n_grid = 1000
    num_params = sum(np.prod(ws, dtype=int) for ws in weight_shapes)

    x_grid = np.linspace(-1.0, 1.0, n_grid)
    mu_grid = np.zeros((n_samples, n_grid))
    y_grid = np.zeros((n_samples, n_grid))
    for ss in xrange(n_samples):
        z_noise = z_std * np.random.randn(num_params)
        mu, prec = primary_out(x_grid[:, None], z_noise)
        std_dev = np.sqrt(1.0 / prec)
        # Just store an MC version than do the mixture of Gauss logic
        mu_grid[ss, :] = mu[:, 0]
        # Note: using same noise across whole grid
        y_grid[ss, :] = mu[:, 0] + std_dev * np.random.randn()

    plt.plot(x_grid, np.mean(mu_grid, axis=0), 'k')
    plt.plot(x_grid, np.percentile(y_grid, 5.0, axis=0), 'k')
    plt.plot(x_grid, np.percentile(y_grid, 95.0, axis=0), 'k')
    plt.plot(x_grid, mu_grid[:5, :].T)
    plt.plot(X, y, '.')
