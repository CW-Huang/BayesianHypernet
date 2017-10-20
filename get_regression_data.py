#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy
np = numpy
np.random.seed(1) # TODO
import os
import random
import pandas as pd

"""
TODO: figure out how to do cross-validation here!
"""


def get_dataset(data, test_inds=None):

	X = data[ :, range(data.shape[ 1 ] - 1) ]
	y = data[ :, data.shape[ 1 ] - 1 ]
	permutation = np.asarray(random.sample(range(0,X.shape[0]), X.shape[0]))

	size_train = int(round(np.round(X.shape[ 0 ] * 0.7)))
	size_valid = int(round(np.round(X.shape[ 0 ] * 0.2)))
	size_test = int(round(np.round(X.shape[ 0 ] * 0.1)))

	# index_train = permutation[ 0 : size_train ]
	# index_valid = permutation[ size_valid : size_test]
	# index_test = permutation[ size_valid : ]

	index_train = permutation[0:size_train]
	index_valid = permutation[size_train+1 : size_train+1+size_valid]
	index_test = permutation[ size_train+1+size_valid : ]

	# index_valid = permutation[355:455]
	# index_test = permutation[455:]

	X_train = X[ index_train, : ]
	y_train = y[ index_train ]

	X_valid = X[index_valid, :]
	y_valid = y[index_valid]

	X_test = X[ index_test, : ]
	y_test = y[ index_test ]

	return X_train, y_train, X_valid, y_valid, X_test, y_test


def get_regression_dataset(dataset_name, test_inds=None):
    if dataset_name == "boston":
    	#(506, 14)
    	data = np.loadtxt('./regression_datasets/boston_housing.txt')

    elif dataset_name == "concrete":
    	#(1029, 9)
    	data = pd.read_csv("./regression_datasets/Concrete_Data.csv")
    	data = np.array(data)

    elif dataset_name == "energy":
    	data = pd.read_csv("./regression_datasets/energy_efficiency.csv")
    	data = np.array(data)
    	data = data[0:766, 0:8]	#### only used this portion of data in related papers (others : NaN)

    elif dataset_name == "kin8nm":
    	data = pd.read_csv("./regression_datasets/kin8nm.csv")
    	data = np.array(data)

    elif dataset_name == "naval":
    	data = np.loadtxt('./regression_datasets/naval_propulsion.txt')

    elif dataset_name == "power":
    	data = pd.read_csv('./regression_datasets/power_plant.csv')
    	data = np.array(data)

    elif dataset_name == "protein":
    	data = pd.read_csv('./regression_datasets/protein_structure.csv')
    	data = np.array(data)

    elif dataset_name == "wine":
    	data = pd.read_csv('./regression_datasets/wineQualityReds.csv')
    	data = np.array(data)

    elif dataset_name == "yacht":
    	data = np.loadtxt('./regression_datasets/yach_data.txt')

    elif dataset_name == "year":
    	raise Exception('Need to process data - convert .txt to .csv')

    else:
    	raise Exception('Need a valid dataset')

    data = data.astype("float32")


    #normalize entire dataset
    #data = (data - data.mean()) / data.var()

    train_x, train_y, valid_x, valid_y, test_x , test_y = get_dataset(data)

    train_y = train_y.reshape((train_y.shape[0],1))
    valid_y = valid_y.reshape((valid_y.shape[0], 1))
    test_y = test_y.reshape((test_y.shape[0],1))

    input_dim = train_x.shape[1]
    

    ###normalizing dataset
    train_x = (train_x - train_x.mean()) / train_x.var()
    train_y = (train_y - train_y.mean()) / train_y.var()

    valid_x = (valid_x - valid_x.mean()) / valid_x.var()
    valid_y = (valid_y - valid_y.mean()) / valid_y.var()


    ## DO NOT normalize test data and test labels
    # test_x = (test_x - test_x.mean()) / test_x.var()
    # test_y = (test_y - test_y.mean()) / test_y.var()


    return input_dim, train_x, train_y, valid_x, valid_y, test_x , test_y



