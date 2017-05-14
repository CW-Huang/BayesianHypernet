import lasagne
import theano
import theano.tensor as T


def log_abs_det_T(W):
    return T.log(T.abs_(T.nlinalg.det(W)))


def hypernet_elbo(X, y, loglik_primary_f, logprior_f, hypernet_f, z_noise, N,
                  log_det_dtheta_dz_f=None):
    assert(X.ndim == 2 and y.ndim == 1)
    assert(z_noise.ndim == 1)

    B = X.shape[0]
    rescale = float(N) / B  # Ensure not integer division

    theta = hypernet_f(z_noise)

    loglik = loglik_primary_f(X, y, theta)
    assert(loglik.ndim == 1)
    loglik_total = T.sum(loglik)

    logprior_theta = logprior_f(theta)
    assert(logprior_f.ndim == 0)

    if log_det_dtheta_dz_f is None: # This is slower, but good for testing
        J = T.jacobian(theta, z_noise)
        penalty = log_abs_det_T(J)
    else:
        penalty = log_det_dtheta_dz_f(z_noise)
    assert(penalty.ndim == 0)

    logprior_z = 0.5 * T.dot(z_noise, z_noise)

    elbo = rescale * loglik_total + logprior_theta + penalty + logprior_z
    return elbo


def build_trainer(phi_shared, N, loglik_primary_f, logprior_f, hypernet_f,
                  log_det_dtheta_dz_f=None):
    '''It is assumed every time this is called z_noise will be drawn from a 
    standard Gaussian. phi_shared are weights to hypernet and N is the total
    number of points in the data set.'''
    X = T.matrix('x')
    y = T.vector('y')
    z_noise = T.vector('z')

    lr = T.scalar('lr')

    elbo = hypernet_elbo(X, y, loglik_primary_f, logprior_f, hypernet_f,
                         z_noise, N, log_det_dtheta_dz_f=log_det_dtheta_dz_f)
    loss = -elbo
    grads = T.grad(loss, phi_shared)
    updates = lasagne.updates.adam(grads, phi_shared, learning_rate=lr)

    trainer = theano.function([X, y, z_noise, lr], loss, updates=updates)
    return trainer
