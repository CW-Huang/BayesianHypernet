#!/usr/bin/env python
import lasagne
import numpy as np
import theano
import theano.tensor as T
floatX = theano.config.floatX


def log_abs_det_T(W):
    # TODO use the version in t_util
    return T.log(T.abs_(T.nlinalg.det(W)))


def hypernet_elbo(X, y, loglik_primary_f, logprior_f, hypernet_f, z_noise, N,
                  log_det_dtheta_dz_f=None, prelim=False):
    assert(X.ndim == 2 and y.ndim == 2)
    assert(z_noise.ndim == 1)

    B = X.shape[0]
    rescale = float(N) / B  # Ensure not integer division

    theta = hypernet_f(z_noise, prelim)

    loglik = loglik_primary_f(X, y, theta)
    assert(loglik.ndim == 1)
    loglik_total = T.sum(loglik)
    assert(loglik_total.ndim == 0)

    logprior_theta = logprior_f(theta)
    assert(logprior_theta.ndim == 0)

    if log_det_dtheta_dz_f is None: # This is slower, but good for testing
        assert(theta.ndim == 1)  # Use vector theta for this mode
        J = T.jacobian(theta, z_noise)
        penalty = log_abs_det_T(J)
    else:
        penalty = log_det_dtheta_dz_f(z_noise, prelim)
    assert(penalty.ndim == 0)

    logprior_z = 0.5 * T.dot(z_noise, z_noise)
    assert(logprior_z.ndim == 0)

    elbo = rescale * loglik_total + logprior_theta + penalty + logprior_z
    return elbo


def build_trainer(phi_shared, N, loglik_primary_f, logprior_f, hypernet_f,
                  log_det_dtheta_dz_f=None, primary_f=None):
    '''It is assumed every time this is called z_noise will be drawn from a 
    standard Gaussian. phi_shared are weights to hypernet and N is the total
    number of points in the data set.'''
    X = T.matrix('x')
    y = T.matrix('y')  # Assuming multivariate output
    z_noise = T.vector('z')
    prelim = T.bscalar('prelim')

    lr = T.scalar('lr')

    elbo = hypernet_elbo(X, y, loglik_primary_f, logprior_f, hypernet_f,
                         z_noise, N, log_det_dtheta_dz_f=log_det_dtheta_dz_f,
                         prelim=prelim)
    loss = -elbo
    grads = T.grad(loss, phi_shared)
    updates = lasagne.updates.adam(grads, phi_shared, learning_rate=lr)

    trainer = theano.function([X, y, z_noise, lr, prelim], loss, updates=updates)

    # Build get_err in case you want to check Jacobian logic
    elbo_no_J = hypernet_elbo(X, y, loglik_primary_f, logprior_f, hypernet_f,
                              z_noise, N, prelim=prelim)
    err = T.abs_(elbo - elbo_no_J)
    get_err = theano.function([X, y, z_noise, prelim], err)

    theta = hypernet_f(z_noise, prelim=prelim)
    test_loglik = loglik_primary_f(X, y, theta)
    test_f = theano.function([X, y, z_noise, prelim], test_loglik)

    primary_out = None
    if primary_f is not None:
        yp = primary_f(X, theta)
        primary_out = theano.function([X, z_noise, prelim], yp)

    grad_f = theano.function([X, y, z_noise, prelim], grads)
    return trainer, get_err, test_f, primary_out, grad_f

# ============================================================================
# Example with learning Gaussian for linear predictor
# ============================================================================


def loglik_primary_f_0(X, y, theta):
    yp = T.dot(X, theta[:, None])
    err = yp - y

    loglik = -0.5 * T.sum(err ** 2, axis=1)  # Ignoring normalizing constant
    return loglik


def logprior_f_0(theta):
    # Standard Gauss
    logprior = -0.5 * T.sum(theta ** 2)  # Ignoring normalizing constant
    return logprior


def simple_test(X, y, n_epochs, n_batch, init_lr, vis_freq=100):
    N, D = X.shape
    assert(y.shape == (N, 1))  # For now assume univariate

    W = theano.shared(np.random.randn(D, D), name='W')
    b = theano.shared(np.random.randn(D), name='b')
    phi_shared = [W, b]

    hypernet_f = lambda z: T.dot(z, W) + b
    log_det_dtheta_dz_f = lambda z: T.log(T.abs_(T.nlinalg.det(W)))
    R = build_trainer(phi_shared, N,
                      loglik_primary_f_0, logprior_f_0, hypernet_f,
                      log_det_dtheta_dz_f=log_det_dtheta_dz_f)
    trainer, get_err, _, _ = R

    t = 0
    for e in range(n_epochs):
        current_lr = init_lr
        for ii in xrange(N / n_batch):
            x_batch = X[ii * n_batch:(ii + 1) * n_batch]
            y_batch = y[ii * n_batch:(ii + 1) * n_batch]

            z_noise = np.random.randn(D).astype(floatX)
            loss = trainer(x_batch, y_batch, z_noise, current_lr)

            if t % vis_freq == 0:
                print 'loss %f' % loss
                err = get_err(x_batch, y_batch, z_noise)
                print 'log10 jac err %f' % np.log10(err)
            t += 1
    return W.get_value(), b.get_value()


if __name__ == '__main__':
    np.random.seed(5645)

    init_lr = 0.0001
    n_epochs = 5000
    n_batch = 32

    D = 5
    N = 1000

    theta_0 = np.random.randn(D, 1).astype(floatX)
    X = np.random.randn(N, D).astype(floatX)
    y = (np.dot(X, theta_0) + np.random.randn(N, 1)).astype(floatX)

    W, b = simple_test(X, y, n_epochs, n_batch, init_lr)
    posterior_cov = np.dot(W.T, W)

    print theta_0[:, 0]
    print b
    print posterior_cov
