#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
from scipy.misc import logsumexp
import scipy.stats as ss
import theano
from theano.ifelse import ifelse
import theano.tensor as T
import np_util
import t_util
from t_util import get_shape

from sklearn.cluster import KMeans

# theano.config.exception_verbosity = 'high'

DEFAULT_VALIDATE_LAYERS = False
summary_f = np.mean

WL_PARAM = 'WL'
bL_PARAM = 'bL'
aL_PARAM = 'aL'

LL_PARAM = 'LL'
UL_PARAM = 'UL'


def get_n_layers(layers, validate=DEFAULT_VALIDATE_LAYERS):
    n_layers, rem = divmod(len(layers) + 1, 3)
    assert(rem == 0)
    assert(n_layers > 0)
    assert((n_layers - 1, aL_PARAM) not in layers)

    if validate and n_layers > 0:
        shape0 = get_shape(layers[(0, WL_PARAM)])
        assert(len(shape0) > 0)
        D = shape0[0]
        for nn in xrange(n_layers):
            assert(get_shape(layers[(nn, WL_PARAM)]) == (D, D))
            assert(get_shape(layers[(nn, bL_PARAM)]) == (D,))
            assert(nn == n_layers - 1 or
                   get_shape(layers[(nn, aL_PARAM)]) == (D,))
    return n_layers

# ============================================================================
# Numpy only section
# ============================================================================


def prelu_np(X, log_alpha):
    act = (X < 0.0) * (np.exp(log_alpha) * X) + (X >= 0.0) * X
    d_log_act = (X < 0.0) * log_alpha
    return act, d_log_act


def LU_to_W_np(layers_LU):
    n_layers, rem = divmod(len(layers_LU) + 1, 4)
    assert(rem == 0)
    assert(n_layers > 0)
    assert((n_layers - 1, aL_PARAM) not in layers_LU)

    layers = layers_LU.copy()
    for nn in xrange(n_layers):
        LL, UL = layers[(nn, LL_PARAM)], layers[(nn, UL_PARAM)]
        layers[(nn, WL_PARAM)] = np.dot(LL, UL)
        del layers[(nn, LL_PARAM)]
        del layers[(nn, UL_PARAM)]
    return layers


def network_np(x, layers):
    assert(x.ndim == 2)
    n_layers = get_n_layers(layers)

    xp = x
    log_d_total = np.zeros(x.shape[0])
    for nn in xrange(n_layers - 1):
        WL, bL = layers[(nn, WL_PARAM)], layers[(nn, bL_PARAM)]
        aL = layers[(nn, aL_PARAM)]
        xp, log_d = prelu_np(np.dot(xp, WL) + bL[None, :], aL[None, :])
        log_d_total = log_d_total + log_d.sum(axis=1)

    # No activation at final step
    WL, bL = layers[(n_layers - 1, WL_PARAM)], layers[(n_layers - 1, bL_PARAM)]
    xp = np.dot(xp, WL) + bL[None, :]
    return xp, log_d_total


def ign_rnd(layers, base_rnd=np.random.randn, N=1, return_hidden=False):
    # TIP could validate here too
    D = layers[(0, WL_PARAM)].shape[0]

    xp = base_rnd(N, D)
    assert(xp.shape == (N, D))
    x, _ = network_np(xp, layers)

    if not return_hidden:
        xp = None
    return x, xp


def ign_log_pdf(X, layers, base_logpdf=ss.norm.logpdf):
    n_layers = get_n_layers(layers)

    # Get Jacobian contribution from WL, known before X
    log_jacobian_WL = 0.0
    for nn in xrange(n_layers):
        WL = layers[(nn, WL_PARAM)]
        log_jacobian_WL = log_jacobian_WL + np.linalg.slogdet(WL)[1]

    # First unwind it to get get original x back (and Jacobian of activations)
    xp, log_act_jac = network_np(X, layers)

    xp_log_pdf = base_logpdf(xp).sum(axis=1)
    # TODO still have option to test this against Theano Jacobian
    jacobian_penalty = log_jacobian_WL + log_act_jac

    x_log_pdf_final = xp_log_pdf + jacobian_penalty
    return x_log_pdf_final, xp


def ign_mix_log_pdf(X, layers, log_weights, base_logpdf=ss.norm.logpdf):
    n_mixtures = len(layers)
    N = X.shape[0]
    assert(log_weights.shape == (n_mixtures,))

    # TODO use logsumexp here too
    log_weights = np.log(np.exp(log_weights) / np.sum(np.exp(log_weights)))

    logpdf = np.zeros((N, n_mixtures))
    for mm in xrange(n_mixtures):
        logpdf[:, mm] = log_weights[mm] + \
                  ign_log_pdf(X, layers[mm], base_logpdf=base_logpdf)[0]
    x_log_pdf_final = logsumexp(logpdf, axis=1)
    return x_log_pdf_final


def build_inv_layers(layers):
    n_layers = get_n_layers(layers)

    inv_layers = {}
    for nn in xrange(n_layers):
        mm = (n_layers - nn) - 1
        WL_new = np.linalg.inv(layers[(mm, WL_PARAM)])
        inv_layers[(nn, WL_PARAM)] = WL_new
        # We could do this more stable with solve, but we have already computed
        # the inverse anyway, so what the hell.
        inv_layers[(nn, bL_PARAM)] = -np.dot(layers[(mm, bL_PARAM)], WL_new)
        if nn < n_layers - 1:
            # alpha parameter in log scale => 1/a is -a
            assert(mm >= 1)
            inv_layers[(nn, aL_PARAM)] = -layers[(mm - 1, aL_PARAM)]
        # else: no activation on final layer
    return inv_layers


def fit_base_layer(x, inf_layers, M=None, eps=0.0, truncate=False):
    n_layers = get_n_layers(inf_layers)
    D = x.shape[1]

    inf_layers = dict(inf_layers)  # Make a copy will modify
    inf_layers[(n_layers - 1, WL_PARAM)] = np.eye(D)
    inf_layers[(n_layers - 1, bL_PARAM)] = np.zeros(D)

    xp, _ = network_np(x, inf_layers)

    if M is None:
        b = np.mean(x, axis=0)
        S = np.cov(x, rowvar=False, bias=True) + eps * np.eye(D)
        L = np.linalg.cholesky(S)
    else:
        b, W, var = np_util.MLE_PCA(xp, M)
        log_epsilon_std = np.log(np.sqrt(var))
        L = np.linalg.cholesky(np_util.gauss_project(W, log_epsilon_std))

        if truncate:
            log_epsilon_std = np.log(np.max(np.diag(L)[M:]))
            L = L - np.exp(log_epsilon_std) * np.eye(D)
            L = (np.concatenate((L[:, :M], np.zeros((D, D - M))), axis=1)
                 + (np.exp(log_epsilon_std) + eps) * np.eye(D))

    WL_new = np.linalg.inv(L.T)
    inf_layers[(n_layers - 1, WL_PARAM)] = WL_new
    inf_layers[(n_layers - 1, bL_PARAM)] = -np.dot(b, WL_new)
    return inf_layers

# ============================================================================
# Theano
# ============================================================================


def prelu_T(X, log_alpha):
    act = T.switch(X < 0.0, T.exp(log_alpha) * X, X)
    log_d_act = T.switch(X < 0.0, log_alpha * T.ones_like(X), T.zeros_like(X))
    return act, log_d_act


def network_T(x_tt, layers):
    assert(x_tt.ndim == 2)
    n_layers = get_n_layers(layers)

    log_d_total = T.zeros((x_tt.shape[0],))
    for nn in xrange(n_layers - 1):
        WL, bL = layers[(nn, WL_PARAM)], layers[(nn, bL_PARAM)]
        aL = layers[(nn, aL_PARAM)]
        x_tt, log_d = prelu_T(T.dot(x_tt, WL) + bL[None, :], aL[None, :])
        log_d_total = log_d_total + log_d.sum(axis=1)
    # No activation at final step
    nn = n_layers - 1
    assert((nn, aL_PARAM) not in layers)
    WL, bL = layers[(nn, WL_PARAM)], layers[(nn, bL_PARAM)]
    x_tt = T.dot(x_tt, WL) + bL[None, :]
    return x_tt, log_d_total


#def diagonalize_network(layers):
#    # TODO test
#    n_layers = get_n_layers(layers)
#
#    layers_diag = layers.copy()
#    for nn in xrange(n_layers):
#        WL = layers[(nn, WL_PARAM)]
#        layers_diag[(nn, WL_PARAM)] = T.nlinalg.extract_diag(WL)
#    return layers_diag
#
#
#def network_T_and_J_diag(X_tt, layers):
#    # TODO test
#    '''diagonalize makes all the W's face zero off diagonal. Sometimes handy
#    for initialization purposes.'''
#    assert(X_tt.ndim == 2)
#
#    layers = diagonalize_network(layers)
#    n_layers = get_n_layers(layers)
#
#    # Get Jacobian contribution from WL, known before X
#    log_jacobian_WL = T.zeros(())
#    for nn in xrange(n_layers):
#        WL = layers[(nn, WL_PARAM)]
#        log_jacobian_WL = log_jacobian_WL + T.sum(T.log(T.abs_(WL)))
#
#    # Now do network_T but with a diagonal W
#    log_d_total = T.zeros((X_tt.shape[0],))
#    for nn in xrange(n_layers - 1):
#        WL, bL = layers[(nn, WL_PARAM)], layers[(nn, bL_PARAM)]
#        aL = layers[(nn, aL_PARAM)]
#        X_tt, log_d = prelu_T(X_tt * WL[None, :] + bL[None, :], aL[None, :])
#        log_d_total = log_d_total + log_d.sum(axis=1)
#    # No activation at final step
#    nn = n_layers - 1
#    assert((nn, aL_PARAM) not in layers)
#    WL, bL = layers[(nn, WL_PARAM)], layers[(nn, bL_PARAM)]
#    xp = X_tt * WL[None, :] + bL[None, :]
#
#    jacobian_penalty = log_jacobian_WL + log_d_total
#    return xp, jacobian_penalty


def init_ign_LU(n_layers, D, aL_val=0.25, WL_val=1e-2):
    layers = {}
    for nn in xrange(n_layers):
        #layers[(nn, LL_PARAM)] = WL_val * np.eye(D) + 1e-2 * WL_val * np.random.randn(D, D)
        #layers[(nn, UL_PARAM)] = np.eye(D) + 1e-2 * WL_val * np.random.randn(D, D)
        layers[(nn, LL_PARAM)] = np.tril(WL_val * np.random.randn(D, D))
        layers[(nn, UL_PARAM)] = np.triu(WL_val * np.random.randn(D, D))
        layers[(nn, bL_PARAM)] = np.zeros(D)
        layers[(nn, aL_PARAM)] = np.log(aL_val * np.ones(D))
    # Can't initialize all at zero to break the symmetry
    layers[(n_layers - 1, bL_PARAM)] = np.random.randn(D)
    del layers[(n_layers - 1, aL_PARAM)]
    return layers


def triangularize_network(layers, force_diag=False):
    n_layers, rem = divmod(len(layers) + 1, 4)
    assert(rem == 0)
    assert(n_layers > 0)
    assert((n_layers - 1, aL_PARAM) not in layers)

    layers_LU = layers.copy()
    for nn in xrange(n_layers):
        LL, UL = layers[(nn, LL_PARAM)], layers[(nn, UL_PARAM)]
        # LL_diag = T.nlinalg.alloc_diag(T.nlinalg.extract_diag(LL))

        #layers_LU[(nn, LL_PARAM)] = \
        #    ifelse(force_diag, LL_diag, T.tril(LL))
        #layers_LU[(nn, UL_PARAM)] = \
        #    ifelse(force_diag, T.eye(UL.shape[0]), T.triu(UL))
        layers_LU[(nn, LL_PARAM)] = \
            ifelse(force_diag, T.tril(LL), T.tril(LL))
        layers_LU[(nn, UL_PARAM)] = \
            ifelse(force_diag, T.triu(UL), T.triu(UL))
    return layers_LU, n_layers


def network_T_and_J_LU(X_tt, layers, force_diag=False):
    # TODO test
    assert(X_tt.ndim == 2)

    layers, n_layers = triangularize_network(layers, force_diag=force_diag)

    # Get Jacobian contribution from WL, known before X
    log_jacobian_WL = T.zeros(())
    for nn in xrange(n_layers):
        LL, UL = layers[(nn, LL_PARAM)], layers[(nn, UL_PARAM)]
        log_jacobian_WL = log_jacobian_WL + t_util.log_abs_det_tri_T(LL)
        log_jacobian_WL = log_jacobian_WL + t_util.log_abs_det_tri_T(UL)

    log_d_total = T.zeros((X_tt.shape[0],))
    for nn in xrange(n_layers - 1):
        LL, UL = layers[(nn, LL_PARAM)], layers[(nn, UL_PARAM)]
        # TODO more efficient version that just mults by x directly
        WL = T.dot(LL, UL)

        bL = layers[(nn, bL_PARAM)]
        aL = layers[(nn, aL_PARAM)]
        X_tt, log_d = prelu_T(T.dot(X_tt, WL) + bL[None, :], aL[None, :])
        log_d_total = log_d_total + log_d.sum(axis=1)
    # No activation at final step
    nn = n_layers - 1
    assert((nn, aL_PARAM) not in layers)
    LL, UL = layers[(nn, LL_PARAM)], layers[(nn, UL_PARAM)]
    # TODO more efficient version that just mults by x directly
    WL = T.dot(LL, UL)
    bL = layers[(nn, bL_PARAM)]
    xp = T.dot(X_tt, WL) + bL[None, :]

    jacobian_penalty = log_jacobian_WL + log_d_total
    return xp, jacobian_penalty


def network_T_and_J(X_tt, layers):
    # TODO test
    assert(X_tt.ndim == 2)
    n_layers = get_n_layers(layers)

    # Get Jacobian contribution from WL, known before X
    log_jacobian_WL = T.zeros(())
    for nn in xrange(n_layers):
        WL = layers[(nn, WL_PARAM)]
        log_jacobian_WL = log_jacobian_WL + t_util.log_abs_det_T(WL)

    # First unwind it to get original x back (and Jacobian of activations)
    xp, log_act_jac = network_T(X_tt, layers)
    jacobian_penalty = log_jacobian_WL + log_act_jac
    return xp, jacobian_penalty


def ign_log_pdf_T(X_tt, layers, base_logpdf=t_util.norm_logpdf_T):
    xp, jacobian_penalty = network_T_and_J(X_tt, layers)
    xp_log_pdf = base_logpdf(xp).sum(axis=1)

    # Final result
    x_log_pdf = xp_log_pdf + jacobian_penalty
    return x_log_pdf, xp


def ign_mix_log_pdf_T(X_tt, layers, log_weights,
                      base_logpdf=t_util.norm_logpdf_T):
    n_mixtures = len(layers)

    log_w_normalized = T.log(T.nnet.softmax(log_weights)[0, :])
    assert(log_w_normalized.ndim == 1)

    # TODO use scan
    loglik_mix = [None] * n_mixtures
    for mm in xrange(n_mixtures):
        loglik, _ = ign_log_pdf_T(X_tt, layers[mm], base_logpdf=base_logpdf)
        assert(loglik.ndim == 1)
        loglik_mix[mm] = log_w_normalized[mm] + loglik
    loglik_mix_T = T.stack(loglik_mix, axis=1)
    # Way to force theano to use logsumexp??
    logpdf = T.log(T.sum(T.exp(loglik_mix_T), axis=1))
    return logpdf, loglik_mix_T


def get_layer_reg_T(layers, reg_dict):
    # TODO formally make connection with normal-wishart
    # Will need to tweak this
    n_layers = get_n_layers(layers)

    reg_cost = 0.0
    # TODO no reg on final layer! check if good idea!
    for nn in xrange(n_layers - 1):
        WL, bL = layers[(nn, WL_PARAM)], layers[(nn, bL_PARAM)]
        sq_logdet = t_util.log_abs_det_T(WL) ** 2
        # Note: no penalty on aL right now!
        reg_cost = reg_cost + (reg_dict[WL_PARAM] * sq_logdet +
                               reg_dict[bL_PARAM] * t_util.L2_T(bL))
    return reg_cost

# ============================================================================
# Wrappers for training
# ============================================================================


def create_ign_updates(x_train, layers, reg_dict, batch_size,
                       base_logpdf=t_util.norm_logpdf_T):
    # Setup preliminary constants
    N, D = x_train.shape
    rescale = N / float(batch_size)

    # Setup shared variables
    x_train = t_util.make_shared(x_train, 'x_train')  # data
    layers_shared = t_util.make_shared_dict(layers, '%d%s')  # parameters

    # Theano input variables (data, etc...)
    x = T.matrix('x')
    epoch = T.scalar('epoch')  # Should this be iscalar??
    batch = T.iscalar('batch')
    lr_decay = T.scalar('lr_decay')

    # Get the likelihood and regularizer to build total cost
    loglik, xp = ign_log_pdf_T(x, layers_shared, base_logpdf=base_logpdf)
    loglik_total = loglik.sum()
    layer_reg_cost = get_layer_reg_T(layers_shared, reg_dict)
    cost = -1.0 * rescale * loglik_total + layer_reg_cost
    cost_unscaled = -1.0 * loglik_total + layer_reg_cost

    # Compute all the gradients, use grad_list to subselect later
    grad_list = layers_shared.values()
    gradients = T.grad(cost, grad_list)

    # Build the update
    updates = t_util.get_adam_min_updates(grad_list, gradients, epoch,
                                          lr_decay=lr_decay)
    # This assumes that batch goes from 0 to int(N / batch_size)-1 inclusive.
    # Under this scheme the remaining partial batch is never used, if the batch
    # number is set to int(N / batch_size) a partial batch at the end is used
    # which results in the rescale variable being the wrong value.
    givens = {x: x_train[batch * batch_size:(batch + 1) * batch_size, :]}
    update = theano.function([batch, epoch, lr_decay], cost,
                             updates=updates, givens=givens)

    # Define a bunch of functions for convenience
    xp_f = theano.function([x], xp)
    loglik_f = theano.function([x], loglik)
    cost_f = theano.function([x], cost_unscaled)
    gradients_f = theano.function([x], gradients)

    # Pack it all up
    state_funcs = (xp_f, loglik_f, cost_f, gradients_f)
    return update, state_funcs, layers_shared


def train_ign(x_train, x_valid, layers, reg_dict, n_epochs, batch_size,
              burn_in=50, base_logpdf=t_util.norm_logpdf_T):
    N, D = x_train.shape
    assert(x_valid.ndim == 2 and x_valid.shape[1] == D)
    batch_order = np.arange(int(N / batch_size))

    R = create_ign_updates(x_train, layers, reg_dict,
                           batch_size=batch_size, base_logpdf=base_logpdf)
    update, state_funcs, layers_shared = R
    _, loglik_f, _, _ = state_funcs

    epoch = 0
    cost_list = np.zeros(n_epochs + 1)
    loglik = np.zeros(n_epochs + 1)
    loglik_valid = np.zeros(n_epochs + 1)
    cost_list[epoch] = np.nan
    loglik[epoch] = loglik_f(x_train).mean()
    loglik_valid[epoch] = summary_f(loglik_f(x_valid))
    print 'initial', loglik[epoch], loglik_valid[epoch]

    best_valid = -np.inf
    while epoch < n_epochs:
        epoch += 1
        np.random.shuffle(batch_order)

        cost = 0.0
        lr_decay = np.sqrt(np.maximum(1.0, epoch - burn_in))
        for batch in batch_order:
            batch_cost = update(batch, epoch, lr_decay)
            cost += batch_cost
        cost /= len(batch_order)
        cost_list[epoch] = cost
        loglik_vec = loglik_f(x_train)
        print ss.describe(loglik_vec)
        loglik[epoch] = loglik_vec.mean()
        loglik_valid[epoch] = summary_f(loglik_f(x_valid))

        # Save whatever had the best validation loss along the way
        if loglik_valid[epoch] > best_valid:
            layers = t_util.make_unshared_dict(layers_shared)
            best_valid = loglik_valid[epoch]

        print 'iter-%d %f %f %f (%f)' % (epoch, cost, loglik[epoch],
                                      loglik_valid[epoch], best_valid)
    assert(epoch == n_epochs)
    print 'best valid %f' % best_valid
    return cost_list, loglik, loglik_valid, layers

# ============================================================================
# Wrappers for training mixture style
# ============================================================================


def create_ign_mix_updates(x_train, layers, log_weights, reg_dict, batch_size,
                           base_logpdf=t_util.norm_logpdf_T):
    # Setup preliminary constants
    N, D = x_train.shape
    n_mixtures = len(layers)
    rescale = N / float(batch_size)
    assert(log_weights.shape == (n_mixtures,))

    # Setup shared variables
    x_train = t_util.make_shared(x_train, 'x_train')  # data
    layers_shared = [t_util.make_shared_dict(ll, str(mm) + '-%d%s')
                     for mm, ll in enumerate(layers)]
    log_weights = t_util.make_shared(log_weights, 'w')
    assert(log_weights.ndim == 1)

    # Theano input variables (data, etc...)
    x = T.matrix('x')
    epoch = T.scalar('epoch')  # Should this be iscalar??
    batch = T.iscalar('batch')
    lr_decay = T.scalar('lr_decay')

    # Get the likelihood and regularizer to build total cost
    loglik, loglik_mix_T = ign_mix_log_pdf_T(x, layers_shared, log_weights,
                                             base_logpdf=base_logpdf)
    loglik_total = loglik.sum()

    reg_cost_total = T.zeros(())
    for mm in xrange(n_mixtures):
        reg_cost = get_layer_reg_T(layers_shared[mm], reg_dict)
        reg_cost_total = reg_cost_total + reg_cost
    cost = -1.0 * rescale * loglik_total + reg_cost_total
    cost_unscaled = -1.0 * loglik_total + reg_cost_total

    # Compute all the gradients, use grad_list to subselect later
    grad_list = sum([ls.values() for ls in layers_shared], [])
    grad_list.append(log_weights)
    gradients = T.grad(cost, grad_list)

    # Build the update
    updates = t_util.get_adam_min_updates(grad_list, gradients, epoch,
                                          lr_decay=lr_decay)
    # This assumes that batch goes from 0 to int(N / batch_size)-1 inclusive.
    # Under this scheme the remaining partial batch is never used, if the batch
    # number is set to int(N / batch_size) a partial batch at the end is used
    # which results in the rescale variable being the wrong value.
    givens = {x: x_train[batch * batch_size:(batch + 1) * batch_size, :]}
    update = theano.function([batch, epoch, lr_decay], cost,
                             updates=updates, givens=givens)

    # Define a bunch of functions for convenience
    xp_f = None  # For now, this is just a place-holder
    loglik_f = theano.function([x], loglik)
    # No rescaling here! Must pass in entire data set to get right answer.
    cost_f = theano.function([x], cost_unscaled)
    gradients_f = theano.function([x], gradients)

    #mix_comp_f = theano.function([x], loglik_mix_T)
    #gradients_mix_comp = T.grad(loglik_mix_T.sum(), grad_list)
    #gradients_dbg_f = theano.function([x], gradients_mix_comp)
    #dbg = (mix_comp_f, gradients_dbg_f)
    dbg = None

    # Pack it all up
    state_funcs = (xp_f, loglik_f, cost_f, gradients_f)
    return update, state_funcs, layers_shared, log_weights, dbg


def init_ign(n_layers, D, aL_val=0.25, rnd_W=True):
    layers = {}
    for nn in xrange(n_layers):
        layers[(nn, WL_PARAM)] = np_util.ortho_rnd(D) if rnd_W else np.eye(D)
        layers[(nn, bL_PARAM)] = np.zeros(D)
        layers[(nn, aL_PARAM)] = np.log(aL_val * np.ones(D))
    del layers[(n_layers - 1, aL_PARAM)]
    return layers


def init_ign_diag(n_layers, D, aL_val=0.25, WL_val=1e-2):
    layers = {}
    for nn in xrange(n_layers):
        layers[(nn, WL_PARAM)] = WL_val * np.eye(D)
        layers[(nn, bL_PARAM)] = np.zeros(D)
        layers[(nn, aL_PARAM)] = np.log(aL_val * np.ones(D))
    # Can't initialize all at zero to break the symmetry
    layers[(n_layers - 1, bL_PARAM)] = np.random.randn(D)
    del layers[(n_layers - 1, aL_PARAM)]
    return layers


def init_ign_mix(x_train, n_layers, n_mixtures, gmm=None, M=10):
    D = x_train.shape[1]

    layers = [None] * n_mixtures
    log_weights = np.zeros(n_mixtures)
    if gmm is None:
        km = KMeans(n_clusters=n_mixtures)
        idx = km.fit_predict(x_train)

        for mm in xrange(len(layers)):
            layers[mm] = init_ign(n_layers, D, rnd_W=True)
            layers[mm] = fit_base_layer(x_train[idx == mm, :], layers[mm], M=M)
            log_weights[mm] = np.log(np.mean(idx == mm))
    else:
        assert(gmm.n_components == n_mixtures)
        log_weights = np.log(gmm.weights_)
        for ii in xrange(n_mixtures):
            mu = gmm.means_[ii, :]
            S = gmm.covariances_[ii, :, :]
            L = np.linalg.cholesky(S)
            WL_new = np.linalg.inv(L.T)

            layers[ii] = init_ign(n_layers, D, aL_val=1.0, rnd_W=False)
            layers[ii][(n_layers - 1, WL_PARAM)] = WL_new
            layers[ii][(n_layers - 1, bL_PARAM)] = -np.dot(mu, WL_new)
    # TIP could do regular train ign for a few iters
    return layers, log_weights


def train_ign_mix(x_train, x_valid, n_layers, n_mixtures, reg_dict, n_epochs,
                  batch_size=20, burn_in=50, gmm=None,
                  base_logpdf=t_util.norm_logpdf_T):
    N, D = x_train.shape
    assert(x_valid.ndim == 2 and x_valid.shape[1] == D)
    batch_order = np.arange(int(N / batch_size))

    layers, log_weights = init_ign_mix(x_train, n_layers, n_mixtures, gmm=gmm)

    R = create_ign_mix_updates(x_train, layers, log_weights, reg_dict,
                               batch_size=batch_size,
                               base_logpdf=base_logpdf)
    update, state_funcs, layers_shared, log_weights_shared, dbg = R
    _, loglik_f, cost_f, gradient_f = state_funcs

    epoch = 0
    cost_list = np.zeros(n_epochs + 1)
    loglik = np.zeros(n_epochs + 1)
    loglik_valid = np.zeros(n_epochs + 1)
    cost_list[epoch] = np.nan
    loglik[epoch] = loglik_f(x_train).mean()
    loglik_valid[epoch] = summary_f(loglik_f(x_valid))
    print 'initial', loglik[epoch], loglik_valid[epoch]

    best_valid = -np.inf
    while epoch < n_epochs:
        epoch += 1
        np.random.shuffle(batch_order)

        cost = 0.0
        lr_decay = 100.0 if epoch <= 10 else np.sqrt(np.maximum(1.0, epoch - burn_in))
        for batch in batch_order:
            # TODO For debug remove!
            #xx = x_train[batch * batch_size:(batch + 1) * batch_size, :]
            #g_list = gradient_f(xx)
            #all_f = all(np.all(np.isfinite(gg)) for gg in g_list)

            batch_cost = update(batch, epoch, lr_decay)
            cost += batch_cost
        cost /= len(batch_order)
        cost_list[epoch] = cost
        #full_cost = cost_f(x_train)
        loglik_vec = loglik_f(x_train)
        print ss.describe(loglik_vec)
        loglik[epoch] = loglik_vec.mean()
        loglik_valid[epoch] = summary_f(loglik_f(x_valid))

        # Save whatever had the best validation loss along the way
        if loglik_valid[epoch] > best_valid:
            layers = [t_util.make_unshared_dict(ll) for ll in layers_shared]
            # TIP should really do a log softmax here too
            log_weights = log_weights_shared.get_value()
            best_valid = loglik_valid[epoch]

        print 'iter-%d %f %f %f (%f)' % (epoch, cost, loglik[epoch],
                                         loglik_valid[epoch], best_valid)
        print '---'
    assert(epoch == n_epochs)
    print 'best valid %f' % best_valid
    return cost_list, loglik, loglik_valid, layers, log_weights

# ============================================================================
# Tests
# ============================================================================


def sample_stable_mat(D, max_cond_number):
    cond_num = np.inf
    assert(cond_num > max_cond_number)
    while cond_num > max_cond_number:
        W = np.random.randn(D, D)
        cond_num = np.linalg.cond(W)
    return W


def run_tests(runs=100, max_cond_number=1e4):
    # TODO write simple test cases for jacobian logic
    err = []
    for _ in xrange(runs):
        D = 1 + np.random.randint(10)
        n_layers = 1 + np.random.randint(4)
        N = 1 + np.random.randint(5)

        # Nonlinear part, possibly more stable to sample inf rather than gen
        inf_layers = {}
        for nn in xrange(n_layers):
            inf_layers[(nn, bL_PARAM)] = np.random.randn(D)
            inf_layers[(nn, WL_PARAM)] = sample_stable_mat(D, max_cond_number)
            if nn < n_layers - 1:
                inf_layers[(nn, aL_PARAM)] = np.random.randn(D)
        gen_layers = build_inv_layers(inf_layers)

        # TODO try other base_rnd
        x, xp0 = ign_rnd(gen_layers, N=N, return_hidden=True)
        xp_, _ = network_np(x, inf_layers)

        x_log_pdf1, xp1 = ign_log_pdf(x, inf_layers)

        x_tt = T.matrix()
        R = ign_log_pdf_T(x_tt, inf_layers)
        test_f = theano.function([x_tt], R)
        x_log_pdf2, xp2 = test_f(x)

        err_curr = (np.max(np.abs(xp_ - xp0)),
                    np.max(np.abs(xp1 - xp0)),
                    np.max(np.abs(xp2 - xp0)),
                    np.max(np.abs(x_log_pdf2 - x_log_pdf1)))
        err.append(err_curr)
        print err_curr
    err = np.array(err)
    print np.log10(err.max(axis=0))


def run_tests_mix(runs=100, max_cond_number=1e4):
    err = []
    for _ in xrange(runs):
        D = 1 + np.random.randint(10)
        n_layers = 1 + np.random.randint(4)
        N = 1 + np.random.randint(5)
        M = 1 + np.random.randint(5)

        # Nonlinear part, possibly more stable to sample inf rather than gen
        inf_layers = [None] * M
        for mm in xrange(M):
            inf_layers[mm] = {}
            for nn in xrange(n_layers):
                inf_layers[mm][(nn, bL_PARAM)] = np.random.randn(D)
                inf_layers[mm][(nn, WL_PARAM)] = sample_stable_mat(D, max_cond_number)
                if nn < n_layers - 1:
                    inf_layers[mm][(nn, aL_PARAM)] = np.random.randn(D)
        log_weights = np.random.randn(M)

        # For now just generate from first mixture
        gen_layers = build_inv_layers(inf_layers[0])

        x, _ = ign_rnd(gen_layers, N=N)

        x_log_pdf1 = ign_mix_log_pdf(x, inf_layers, log_weights)

        x_tt = T.matrix()
        R = ign_mix_log_pdf_T(x_tt, inf_layers, log_weights)
        test_f = theano.function([x_tt], R)
        x_log_pdf2 = test_f(x)

        err_curr = (np.max(np.abs(x_log_pdf2 - x_log_pdf1)),)
        err.append(err_curr)
        print err_curr
    err = np.array(err)
    print np.log10(err.max(axis=0))

if __name__ == '__main__':
    np.random.seed(74100)
    run_tests()
