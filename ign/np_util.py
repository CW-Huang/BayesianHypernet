#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import scipy.linalg as sl


def MLE_PCA(X, q):
    N, D = X.shape
    assert(q <= N)

    mu = np.mean(X, axis=0)

    S = np.cov(X, rowvar=False, bias=True)
    Lam, U = np.linalg.eigh(S)

    var = np.mean(Lam[:-q])
    W = np.dot(U[:, -q:], np.diag(np.sqrt(Lam[-q:] - var))).T
    return mu, W, var


def gauss_project(W, log_epsilon_std):
    assert(W.ndim == 2 and np.ndim(log_epsilon_std) == 0)
    D = W.shape[1]
    S = np.dot(W.T, W) + (np.exp(log_epsilon_std) ** 2) * np.eye(D)
    return S


def rel_err(v1, v2):
    re = np.abs(v1 - v2) / np.minimum(np.abs(v1), np.abs(v1))
    return re


def skew_sym_rnd(D):
    L = np.tril(np.random.randn(D, D), k=-1)
    S = L - L.T
    return S


def ortho_rnd(D):
    A = skew_sym_rnd(D)
    Q = sl.solve(np.eye(D) - A, np.eye(D) + A)
    return Q
