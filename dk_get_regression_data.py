#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy
np = numpy
np.random.seed(1) # TODO
import os
import random



def get_regression_dataset(dataset, split_count, data_path='./', valid_set=True, normalize=True):#$HOME/BayesianHypernetCW/'):
    # load data
    data = np.loadtxt(data_path + 'get_proper_regression_data/' + dataset + '/data/data.txt').astype('float32')
    index_features = np.loadtxt(data_path + 'get_proper_regression_data/' + dataset + '/data/index_features.txt')
    index_target = np.loadtxt(data_path + 'get_proper_regression_data/' + dataset + '/data/index_target.txt')
    index_train = np.loadtxt(data_path + "get_proper_regression_data/" + dataset + "/data/index_train_{}.txt".format(split_count))
    index_test = np.loadtxt(data_path + "get_proper_regression_data/" + dataset + "/data/index_test_{}.txt".format(split_count))

    # splitting and whatnot
    X = data[:, index_features.astype(int)]
    y = data[:, index_target.astype(int)].reshape((-1,1))
    train_x, train_y, test_x , test_y = get_dataset(X, y, split_count, index_train, index_test)
    input_dim = train_x.shape[1]

    if valid_set:
        ind = int(.8 * len(train_x))
        valid_x = train_x[ind:]
        valid_y = train_y[ind:]
        train_x = train_x[:ind]
        train_y = train_y[:ind]

        if normalize:
            x_mean = train_x.mean(axis=0)
            x_std = train_x.std(axis=0)
            y_mean = train_y.mean()
            y_std = train_y.std()

            x_std[ x_std == 0 ] = 1 # avoid divide by 0!

            train_x = (train_x - x_mean) / x_std
            valid_x = (valid_x - x_mean) / x_std
            test_x = (test_x - x_mean) / x_std

            train_y = (train_y - y_mean) / y_std

            # TODO: these must be "unnormalized" by the model
            #valid_y = (valid_y - y_mean) / y_std
            #test_y = (test_y - y_mean) / y_std
        else:
            y_mean = None
            y_std = None

        return input_dim, train_x, train_y, valid_x, valid_y, test_x, test_y, y_mean, y_std
    else:
        return input_dim, train_x, train_y, test_x, test_y


def get_dataset(X, y, i, index_train, index_test):

    X_train = X[ index_train.astype(int), ]
    y_train = y[ index_train.astype(int) ]
    X_test = X[ index_test.astype(int), ]
    y_test = y[ index_test.astype(int) ]

    return X_train, y_train, X_test, y_test
