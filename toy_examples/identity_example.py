#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)
from collections import OrderedDict
import cPickle as pkl
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy.stats import gaussian_kde
import mlp_hmc


def just_kde(x, y, n_grid_x=1000, n_grid_y=1000, extent=None):
    assert(y.ndim == 1 and x.shape == y.shape)

    data = np.vstack((x, y))  # 2 x N
    kernel = gaussian_kde(data)

    if extent is None:
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
    x_grid = np.linspace(extent[0], extent[1], n_grid_x)
    y_grid = np.linspace(extent[2], extent[3], n_grid_y)

    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    Z = np.reshape(kernel(positions).T, xx.shape)
    assert(Z.shape == (n_grid_y, n_grid_x))

    return Z, extent


def multi_kde(x_list, y_list, n_grid_x=1000, n_grid_y=1000):
    '''Perform multiple kde but with axis limits syncronized.'''
    extent = [np.min(x_list), np.max(x_list), np.min(y_list), np.max(y_list)]

    Z = [just_kde(x, y, n_grid_x=n_grid_x, n_grid_y=n_grid_y, extent=extent)[0]
         for x, y in zip(x_list, y_list)]
    return Z, extent


def run_id_experiment():
    np.random.seed(75674)
    test_run = False

    n_tune_hmc = 5 if test_run else 50
    n_iter_hmc = 3 if test_run else 50
    n_samples = 5 if test_run else 100

    num_params = 3
    weight_shapes = OrderedDict([('a', ()), ('b', ()), ('c', ())])
    assert(num_params == mlp_hmc.get_num_params(weight_shapes))

    df = pd.read_csv('samples.csv', header=0, index_col=None)
    df = df[weight_shapes.keys()]  # Make sure we are in same order
    init_samples = df.values

    df = pd.read_csv('data_train.csv', header=0, index_col=None)
    X = df['x'].values
    y = df['y'].values

    df = pd.read_csv('data_valid.csv', header=0, index_col=None)
    X_valid = df['x'].values
    y_valid = df['y'].values

    # Get HMC result
    tr_list, hmc_dbg = \
        mlp_hmc.prod_net_hmc(X, y, X_valid, y_valid, init_samples,
                             weight_shapes, restarts=n_samples,
                             n_iter=n_iter_hmc, n_tune=n_tune_hmc)
    a_hmc = np.concatenate([pm.get_values('a') for tr in tr_list], axis=0)
    b_hmc = np.concatenate([pm.get_values('b') for tr in tr_list], axis=0)
    assert(a_hmc.ndim == 1)
    assert(a_hmc.shape == b_hmc.shape)

    # Get hypernet results
    theta = [mlp_hmc.unpack(v) for v in init_samples]
    a_hyper = np.array([D['a'] for D in theta])
    b_hyper = np.array([D['b'] for D in theta])
    assert(a_hyper.ndim == 1)
    assert(a_hyper.shape == b_hyper.shape)

    (Z_hyper, Z_hmc), extent = multi_kde([a_hyper, a_hmc], [b_hyper, b_hmc])

    dump_dict = {}
    dump_dict['data'] = X, y
    dump_dict['hmc'] = a_hmc, b_hmc
    dump_dict['hmc_dbg'] = hmc_dbg
    dump_dict['hmc_density'] = Z_hmc, extent
    dump_dict['hyper'] = a_hyper, b_hyper
    dump_dict['hyper_density'] = Z_hyper, extent
    with open('id_example_dump.pkl', 'wb') as f:
        pkl.dump(dump_dict, f, protocol=0)

if __name__ == '__main__':
    run_id_experiment()
