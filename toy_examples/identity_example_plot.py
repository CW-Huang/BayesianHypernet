#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import numpy as np

from matplotlib import rcParams, use
# use('pdf')
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt


def plot_case(ax, Z, extent):
    x_grid = np.linspace(extent[0], extent[1], 1000)

    plt.imshow(Z, extent=extent)
    plt.plot(x_grid, 1.0 / x_grid, 'r')
    ax.grid()


def plot_dump(D):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True,
                                   figsize=(6.0, 3.5), dpi=300,
                                   facecolor='w', edgecolor='k')

    Z, extent = D['hyper_density']
    plot_case(ax1, Z, extent)
    ax1.set_title('hypernet', fontsize=10)
    ax1.tick_params(labelsize=8)

    Z, extent = D['hmc_density']
    plot_case(ax2, Z, extent)
    ax2.set_title('NUTS', fontsize=10)
    ax2.tick_params(labelsize=8)

    return fig, (ax1, ax2)

if __name__ == '__main__':
    fname = 'id_example_dump.pkl'
    print fname
    with open(fname, 'rb') as f:
        D = pkl.load(f)
    fig, (ax1, ax2) = plot_dump(D)
    print 'done'
