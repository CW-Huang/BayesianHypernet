#!/usr/bin/env python
from ops import load_mnist
from utils import log_normal, log_laplace
import numpy
np = numpy
import random
random.seed(5001)
from lasagne import layers
from scipy.stats import mode

import time
import os


def to_categorical(y):
    num_classes=10
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1

    return categorical


def split_train_pool_data(X_train, y_train):

    X_train_All = X_train
    y_train_All = y_train

    random_split = np.asarray(random.sample(range(0,X_train_All.shape[0]), X_train_All.shape[0]))

    X_train_All = X_train_All[random_split, :, :, :]
    y_train_All = y_train_All[random_split]

    X_pool = X_train_All[10000:50000, :, :, :]
    y_pool = y_train_All[10000:50000]


    X_train = X_train_All[0:10000, :, :, :]
    y_train = y_train_All[0:10000]


    return X_train, y_train, X_pool, y_pool

def get_initial_training_data(X_train_All, y_train_All):
    #training data to have equal distribution of classes
    idx_0 = np.array( np.where(y_train_All==0)  ).T
    idx_0 = idx_0[0:2,0]
    X_0 = X_train_All[idx_0, :, :, :]
    y_0 = y_train_All[idx_0]

    idx_1 = np.array( np.where(y_train_All==1)  ).T
    idx_1 = idx_1[0:2,0]
    X_1 = X_train_All[idx_1, :, :, :]
    y_1 = y_train_All[idx_1]

    idx_2 = np.array( np.where(y_train_All==2)  ).T
    idx_2 = idx_2[0:2,0]
    X_2 = X_train_All[idx_2, :, :, :]
    y_2 = y_train_All[idx_2]

    idx_3 = np.array( np.where(y_train_All==3)  ).T
    idx_3 = idx_3[0:2,0]
    X_3 = X_train_All[idx_3, :, :, :]
    y_3 = y_train_All[idx_3]

    idx_4 = np.array( np.where(y_train_All==4)  ).T
    idx_4 = idx_4[0:2,0]
    X_4 = X_train_All[idx_4, :, :, :]
    y_4 = y_train_All[idx_4]

    idx_5 = np.array( np.where(y_train_All==5)  ).T
    idx_5 = idx_5[0:2,0]
    X_5 = X_train_All[idx_5, :, :, :]
    y_5 = y_train_All[idx_5]

    idx_6 = np.array( np.where(y_train_All==6)  ).T
    idx_6 = idx_6[0:2,0]
    X_6 = X_train_All[idx_6, :, :, :]
    y_6 = y_train_All[idx_6]

    idx_7 = np.array( np.where(y_train_All==7)  ).T
    idx_7 = idx_7[0:2,0]
    X_7 = X_train_All[idx_7, :, :, :]
    y_7 = y_train_All[idx_7]

    idx_8 = np.array( np.where(y_train_All==8)  ).T
    idx_8 = idx_8[0:2,0]
    X_8 = X_train_All[idx_8, :, :, :]
    y_8 = y_train_All[idx_8]

    idx_9 = np.array( np.where(y_train_All==9)  ).T
    idx_9 = idx_9[0:2,0]
    X_9 = X_train_All[idx_9, :, :, :]
    y_9 = y_train_All[idx_9]

    X_train = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0 )
    y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0 )
    
    y_train = to_categorical(y_train)

    return X_train, y_train



