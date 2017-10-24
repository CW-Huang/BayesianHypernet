#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy
np = numpy
np.random.seed(1) # TODO
import os
import random
import pandas as pd



def get_regression_dataset(dataset, split_count, data_path='./'):#$HOME/BayesianHypernetCW/'):

    if dataset == "boston":

        data = np.loadtxt(data_path + 'get_proper_regression_data/bostonHousing/data/data.txt')
        index_features = np.loadtxt('get_proper_regression_data/bostonHousing/data/index_features.txt')
        index_target = np.loadtxt('get_proper_regression_data/bostonHousing/data/index_target.txt')

        index_train = np.loadtxt(data_path + "get_proper_regression_data/bostonHousing/data/index_train_{}.txt".format(split_count))
        index_test = np.loadtxt(data_path + "get_proper_regression_data/bostonHousing/data/index_test_{}.txt".format(split_count))


    elif dataset == "concrete":
        data = np.loadtxt(data_path + 'get_proper_regression_data/concrete/data/data.txt')
        index_features = np.loadtxt('get_proper_regression_data/concrete/data/index_features.txt')
        index_target = np.loadtxt('get_proper_regression_data/concrete/data/index_target.txt')

        index_train = np.loadtxt(data_path + "get_proper_regression_data/concrete/data/index_train_{}.txt".format(split_count))
        index_test = np.loadtxt(data_path + "get_proper_regression_data/concrete/data/index_test_{}.txt".format(split_count))


    elif dataset == "energy":
        data = np.loadtxt(data_path + 'get_proper_regression_data/energy/data/data.txt')
        index_features = np.loadtxt('get_proper_regression_data/energy/data/index_features.txt')
        index_target = np.loadtxt('get_proper_regression_data/energy/data/index_target.txt')

        index_train = np.loadtxt(data_path + "get_proper_regression_data/energy/data/index_train_{}.txt".format(split_count))
        index_test = np.loadtxt(data_path + "get_proper_regression_data/energy/data/index_test_{}.txt".format(split_count))



    elif dataset == "kin8nm":
        data = np.loadtxt(data_path + 'get_proper_regression_data/kin8nm/data/data.txt')
        index_features = np.loadtxt('get_proper_regression_data/kin8nm/data/index_features.txt')
        index_target = np.loadtxt('get_proper_regression_data/kin8nm/data/index_target.txt')

        index_train = np.loadtxt(data_path + "get_proper_regression_data/kin8nm/data/index_train_{}.txt".format(split_count))
        index_test = np.loadtxt(data_path + "get_proper_regression_data/kin8nm/data/index_test_{}.txt".format(split_count))



    elif dataset == "naval-propulsion-plant":
        data = np.loadtxt(data_path + 'get_proper_regression_data/naval-propulsion-plant/data/data.txt')
        index_features = np.loadtxt('get_proper_regression_data/naval-propulsion-plant/data/index_features.txt')
        index_target = np.loadtxt('get_proper_regression_data/naval-propulsion-plant/data/index_target.txt')

        index_train = np.loadtxt(data_path + "get_proper_regression_data/naval-propulsion-plant/data/index_train_{}.txt".format(split_count))
        index_test = np.loadtxt(data_path + "get_proper_regression_data/naval-propulsion-plant/data/index_test_{}.txt".format(split_count))



    elif dataset == "power":
        data = np.loadtxt(data_path + 'get_proper_regression_data/power-plant/data/data.txt')        
        index_features = np.loadtxt('get_proper_regression_data/power-plant/data/index_features.txt')
        index_target = np.loadtxt('get_proper_regression_data/power-plant/data/index_target.txt')

        index_train = np.loadtxt(data_path + "get_proper_regression_data/power-plant/data/index_train_{}.txt".format(split_count))
        index_test = np.loadtxt(data_path + "get_proper_regression_data/power-plant/data/index_test_{}.txt".format(split_count))


    elif dataset == "protein":
        data = np.loadtxt(data_path + 'get_proper_regression_data/protein-tertiary-structure/data/data.txt')   
        index_features = np.loadtxt('get_proper_regression_data/protein-tertiary-structure/data/index_features.txt')
        index_target = np.loadtxt('get_proper_regression_data/protein-tertiary-structure/data/index_target.txt')

        index_train = np.loadtxt(data_path + "get_proper_regression_data/protein-tertiary-structure/data/index_train_{}.txt".format(split_count))
        index_test = np.loadtxt(data_path + "get_proper_regression_data/protein-tertiary-structure/data/index_test_{}.txt".format(split_count))


    elif dataset == "wine":
        data = np.loadtxt(data_path + 'get_proper_regression_data/wine-quality-red/data/data.txt')  
        index_features = np.loadtxt('get_proper_regression_data/wine-quality-red/data/index_features.txt')
        index_target = np.loadtxt('get_proper_regression_data/wine-quality-red/data/index_target.txt')

        index_train = np.loadtxt(data_path + "get_proper_regression_data/wine-quality-red/data/index_train_{}.txt".format(split_count))
        index_test = np.loadtxt(data_path + "get_proper_regression_data/wine-quality-red/data/index_test_{}.txt".format(split_count))


    elif dataset == "yacht":
        data = np.loadtxt(data_path + 'get_proper_regression_data/yacht/data/data.txt')  
        index_features = np.loadtxt('get_proper_regression_data/yacht/data/index_features.txt')
        index_target = np.loadtxt('get_proper_regression_data/yacht/data/index_target.txt')

        index_train = np.loadtxt(data_path + "get_proper_regression_data/yacht/data/index_train_{}.txt".format(split_count))
        index_test = np.loadtxt(data_path + "get_proper_regression_data/yacht/data/index_test_{}.txt".format(split_count))


    elif dataset == "year":
        raise Exception('Dataset too big - not available in repo')

    else :
        raise Exception('Need a valid dataset')


    X = data[:, index_features.astype(int)]
    y = data[:, index_target.astype(int)]

    # n_splits = 20
    ### split count - from 0 - 19
    train_x, train_y, test_x , test_y = get_dataset(X, y, split_count, index_train, index_test)

    input_dim = train_x.shape[1]

    return input_dim, train_x, train_y, test_x , test_y


def get_dataset(X, y, i, index_train, index_test):

    X_train = X[ index_train.astype(int), ]
    y_train = y[ index_train.astype(int) ]
    X_test = X[ index_test.astype(int), ]
    y_test = y[ index_test.astype(int) ]


    std_X_train = np.std(X_train, 0)
    std_X_train[ std_X_train == 0 ] = 1
    mean_X_train = np.mean(X_train, 0)

    X_train = (X_train - np.full(X_train.shape, mean_X_train)) / \
    np.full(X_train.shape, std_X_train)


    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)


    y_train_normalized = (y_train - mean_y_train) / std_y_train
    y_train_normalized = np.array(y_train_normalized, ndmin = 2).T

    y_train = y_train_normalized


    X_test = np.array(X_test, ndmin = 2)
    y_test = np.array(y_test, ndmin = 2).T

    # We normalize the test set
    X_test = (X_test - np.full(X_test.shape, mean_X_train)) / np.full(X_test.shape, std_X_train)

    return X_train, y_train, X_test, y_test




