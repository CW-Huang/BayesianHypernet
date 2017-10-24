#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import numpy as np

from matplotlib import rcParams, use
# use('pdf')
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt

STD_GRID = (0.5, 1.0, 1.5, 2.0)
N_POINTS = 50


def plot_case(ax, x, y, x_grid, mu, std, color='b', alpha=0.3):
    assert(x_grid.ndim == 1)
    assert(mu.shape == x_grid.shape)
    assert(std.shape == mu.shape)
    assert(np.all(std >= 0.0))

    for fac in STD_GRID:
        LB = mu - fac * std
        UB = mu + fac * std
        ax.fill_between(x_grid, LB, UB, color=color, alpha=alpha)
    ax.plot(x, y, 'rx')
    ax.plot(x_grid, mu, 'k')
    ax.grid()
    ax.set_xlim(x_grid[0], x_grid[-1])


def plot_dump(D):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True,
                                        figsize=(6.0, 3.5), dpi=300,
                                        facecolor='w', edgecolor='k')

    X, y = D['data']
    X = X[:50, 0]
    y = y[:50, 0]

    mu, std = D['trad']
    plot_case(ax1, X, y, D['x'], mu, std)
    ax1.set_title('traditional', fontsize=10)
    ax1.tick_params(labelsize=8)

    mu, std, _, _, _ = D['hyper']
    plot_case(ax2, X, y, D['x'], mu, std)
    ax2.set_title('hypernet', fontsize=10)
    ax2.tick_params(labelsize=8)

    mu, std, _, _, _ = D['hmc']
    plot_case(ax3, X, y, D['x'], mu[-1, :], std[-1, :])
    ax3.set_title('NUTS', fontsize=10)
    ax3.tick_params(labelsize=8)

    return fig, (ax1, ax2, ax3)

if __name__ == '__main__':
    fname = 'reg_example_dump.pkl'
    print fname
    with open(fname, 'rb') as f:
        D = pkl.load(f)
    fig, (ax1, ax2, ax3) = plot_dump(D)
    print 'done'
