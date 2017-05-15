# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import theano.tensor as T
import hypernet_trainer as ht
import ign.ign as ign
from ign.t_util import make_shared_dict


def dm_example(N):
    x = 0.5 * np.random.rand(N, 1)

    std_dev = np.sqrt(0.02)
    noise = std_dev * np.random.randn(N, 1)
    x_n = x + noise
    y = x + 0.3 * np.sin(2.0 * np.pi * x_n) + 0.3 * np.sin(4.0 * np.pi * x_n) \
        + noise
    return x, y


def unpack(v, weight_shapes):
    L = []
    tt = 0
    for ws in weight_shapes:
        num_param = np.prod(ws)
        L.append(v[tt:tt + num_param].reshape(ws))
        tt += num_param
    return L


def loglik_primary_f(X, y, theta, weight_shapes):
    W0, b0, W1, b1 = unpack(theta, weight_shapes)

    a1 = T.maximum(0.0, T.dot(X, W0) + b0[None, :])
    yp = T.dot(a1, W1) + b1[None, :]
    err = yp - y

    # Assuming std=1 here
    loglik = -0.5 * T.sum(err ** 2, axis=1)  # Ignoring normalizing constant
    return loglik


def logprior_f(theta):
    # Standard Gauss
    logprior = -0.5 * T.sum(theta ** 2)  # Ignoring normalizing constant
    return logprior


def simple_test(X, y, n_epochs, n_batch, init_lr, weight_shapes,
                n_layers=5, vis_freq=100):
    N, D = X.shape
    assert(y.shape == (N, 1))  # Univariate for now
    num_params = sum(np.prod(ws) for ws in weight_shapes)

    layers = ign.init_ign(n_layers, num_params)
    phi_shared = make_shared_dict(layers, '%d%s')

    ll_primary_f = lambda X, y, w: loglik_primary_f(X, y, w, weight_shapes)
    hypernet_f = lambda z: ign.network_T_and_J(z[None, :], phi_shared)[0][0, :]
    # TODO verify this length of size 1
    log_det_dtheta_dz_f = lambda z: T.sum(ign.network_T_and_J(z[None, :], phi_shared)[1])

    R = ht.build_trainer(phi_shared.values(), N, ll_primary_f, logprior_f, hypernet_f,
                         log_det_dtheta_dz_f=log_det_dtheta_dz_f)
    trainer, get_err = R

    t = 0
    for e in range(n_epochs):
        current_lr = init_lr
        for ii in xrange(N / n_batch):
            x_batch = X[ii * n_batch:(ii + 1) * n_batch]
            y_batch = y[ii * n_batch:(ii + 1) * n_batch]

            z_noise = np.random.randn(num_params)
            loss = trainer(x_batch, y_batch, z_noise, current_lr)

            if t % vis_freq == 0:
                print 'loss %f' % loss
                err = get_err(x_batch, y_batch, z_noise)
                print 'log10 jac err %f' % np.log10(err)
            t += 1
    return phi_shared


if __name__ == '__main__':
    np.random.seed(5645)

    init_lr = 0.001
    n_epochs = 500
    n_batch = 32
    N = 1000

    primary_layers = 1
    input_dim = 1
    hidden_dim = 10
    output_dim = 1

    weight_shapes = [(input_dim, hidden_dim), (hidden_dim,),
                     (hidden_dim, output_dim), (output_dim,)]

    X, y = dm_example(N)
    phi = simple_test(X, y, n_epochs, n_batch, init_lr, weight_shapes,
                      n_layers=5)
