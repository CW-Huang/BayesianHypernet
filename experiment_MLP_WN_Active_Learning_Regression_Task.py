import math
import numpy as np
import random
np.random.seed(1)
import pandas as pd

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: Riashat Islam
"""

from BHNs_MLP_Regression import MLPWeightNorm_BHN, MCdropout_MLP
from ops import load_mnist
from utils import log_normal, log_laplace
import numpy as np

import lasagne
import theano
import theano.tensor as T
import os
from lasagne.random import set_rng
from theano.tensor.shared_randomstreams import RandomStreams
from scipy.stats import entropy

lrdefault = 1e-3    
    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def train_model(train_func,predict_func,X,Y,Xv,Yv,
                lr0=0.1,lrdecay=1,bs=16,epochs=50,anneal=0,name='0',
                e0=0,rec=0):
    
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
    va_rec_name = name+'_recs'
    save_path = name + '.params'
    va_recs = list()
    tr_recs = list()
    
    t = 0
    for e in range(epochs):
        
        if e <= e0:
            continue
        
        if lrdecay:
            lr = lr0 * 10**(-e/float(epochs-1))
        else:
            lr = lr0         
        
        if anneal:
            w = min(1.0,0.001+e/(epochs/2.))
        else:
            w = 1.0         
            
        for i in range(N/bs):
            x = X[i*bs:(i+1)*bs]
            y = Y[i*bs:(i+1)*bs]

            loss = train_func(x,y,N,lr,w)
            print ("Loss", loss)

            if t%100==0:
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)

                tr_rmse = rmse(predict_func(X), Y)
                va_rmse = rmse(predict_func(Xv), Yv)
                # print '\ttrain rmse: {}'.format(tr_rmse)
                # print '\tvalid rmse: {}'.format(va_rmse)
            t+=1


        tr_rmse = evaluate_model(model.predict, X, Y, n_mc=20)   
        print '\n\ntr rmse at epochs {}: {}'.format(e,tr_rmse)    
        va_rmse = evaluate_model(model.predict,Xv,Yv,n_mc=20)
        print '\n\nva rmse at epochs {}: {}'.format(e,va_rmse)    
        
        va_recs.append(va_rmse)
        tr_recs.append(tr_rmse)
        
        if va_rmse > rec:
            print '.... save best model .... '
            model.save(save_path,[e])
            rec = va_rmse
    
            with open(va_rec_name,'a') as rec_file:
                for r in va_recs:
                    rec_file.write(str(r)+'\n')
            
            va_recs = list()
            
        print '\n\n'

    validation_rmse = np.asarray(va_recs)
    training_rmse = np.asarray(tr_recs)

    training_rmse_with_current_data = training_rmse[-1]
    validation_rmse_with_current_data = validation_rmse[-1]



    return training_rmse, validation_rmse, training_rmse_with_current_data, validation_rmse_with_current_data
    


def evaluate_model(predict_proba,X,Y,n_mc=100,max_n=100):
    MCt = np.zeros((n_mc,X.shape[0],1))
    
    N = X.shape[0]
    num_batches = np.ceil(N / float(max_n)).astype(int)
    for i in range(n_mc):
        for j in range(num_batches):
            x = X[j*max_n:(j+1)*max_n]
            MCt[i,j*max_n:(j+1)*max_n] = predict_proba(x)

    Y_pred = MCt.mean(0)
    Y_true = Y
    RMSE = rmse(Y_pred, Y_true)

    return RMSE


def get_dataset(data):

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


def get_active_learning_dataset_split(train_x, train_y, valid_x, valid_y, test_x, test_y):

    pool_x = train_x[21:, :]
    pool_y = train_y[21:, :]

    train_x = train_x[0:20, :]
    train_y = train_y[0:20, :]

    valid_x = valid_x[0:100, :]
    valid_y = valid_y[0:100, :]

    test_x = test_x[0:100, :]
    test_y = test_y[0:100, :]

    return pool_x, pool_y, train_x, train_y, valid_x, valid_y, test_x, test_y



def active_learning_acquisition_bald(model_prediction, pool_x, pool_y, train_x, train_y):

    mc_samples = 100
    Queries = 1
    #all_stochastic_y_preds = np.zeros(shape=(pool_x.shape[0], mc_samples))

    score_All = np.zeros(shape=(pool_x.shape[0], 1))
    All_Entropy_Stochastic = np.zeros(shape=pool_x.shape[0])

    for m in range(mc_samples):
        stochastic_y_preds = model_prediction(pool_x)
        #all_stochastic_y_preds[:, m] = stochastic_y_preds[:,0]   #we need this to compute mean/variance (not needed here)
        score_All = score_All + stochastic_y_preds 


        entropy_stochastic_prediction = entropy(stochastic_y_preds)
        sum_all_entropy_stochastic_prediction = All_Entropy_Stochastic + entropy_stochastic_prediction


    avg_stochastic_preds = np.divide(score_All, mc_samples)
    entropy_of_average = entropy(avg_stochastic_preds)

    avg_entropy = np.divide(sum_all_entropy_stochastic_prediction, mc_samples)

    U_X = entropy_of_average - avg_entropy

    a_1d = U_X.flatten()
    x_pool_index = a_1d.argsort()[-Queries:][::-1]

    queried_x = pool_x[x_pool_index]
    queried_y = pool_y[x_pool_index]


    #delete point from pool set
    pool_x = np.delete(pool_x,(x_pool_index), axis=0)
    pool_y = np.delete(pool_y, (x_pool_index), axis=0)

    #add queried point to training set
    train_x = np.concatenate((train_x, queried_x), axis=0)
    train_y = np.concatenate((train_y, queried_y), axis=0)



    return pool_x, pool_y, train_x, train_y



def active_learning_acquisition_highest_entropy(model_prediction, pool_x, pool_y, train_x, train_y):

    mc_samples = 100
    Queries = 1
    all_stochastic_y_preds = np.zeros(shape=(pool_x.shape[0], 1))

    score_All = np.zeros(shape=(pool_x.shape[0], 1))
    All_Entropy_Stochastic = np.zeros(shape=pool_x.shape[0])

    for m in range(mc_samples):
        stochastic_y_preds = model_prediction(pool_x)
        all_stochastic_y_preds = all_stochastic_y_preds + stochastic_y_preds

        
    Avg_Pi = np.divide(all_stochastic_y_preds, mc_samples)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)


    U_X = Entropy_Avg_Pi
    a_1d = U_X.flatten()
    x_pool_index = a_1d.argsort()[-Queries:][::-1]

    queried_x = pool_x[x_pool_index]
    queried_y = pool_y[x_pool_index]


    #delete point from pool set
    pool_x = np.delete(pool_x,(x_pool_index), axis=0)
    pool_y = np.delete(pool_y, (x_pool_index), axis=0)

    #add queried point to training set
    train_x = np.concatenate((train_x, queried_x), axis=0)
    train_y = np.concatenate((train_y, queried_y), axis=0)

    return pool_x, pool_y, train_x, train_y



def active_learning_acquisition_highest_variance(model_prediction, pool_x, pool_y, train_x, train_y):

    mc_samples = 100
    Queries = 1

    all_stochastic_y_preds = np.zeros(shape=(pool_x.shape[0], mc_samples))
    Variance = np.zeros(shape=(pool_x.shape[0]))


    for m in range(mc_samples):
        stochastic_y_preds = model_prediction(pool_x)
        all_stochastic_y_preds[:, m] = stochastic_y_preds[:,0]


    for j in range(pool_x.shape[0]):
        L = all_stochastic_y_preds[j, :]
        L_var = np.var(L)
        Variance[j] = L_var

    v_sort = Variance.flatten()
    x_pool_index = v_sort.argsort()[-Queries:][::-1]

    queried_x = pool_x[x_pool_index]
    queried_y = pool_y[x_pool_index]


    #delete point from pool set
    pool_x = np.delete(pool_x,(x_pool_index), axis=0)
    pool_y = np.delete(pool_y, (x_pool_index), axis=0)

    #add queried point to training set
    train_x = np.concatenate((train_x, queried_x), axis=0)
    train_y = np.concatenate((train_y, queried_y), axis=0)

    return pool_x, pool_y, train_x, train_y








if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--lrdecay',default=0.1,type=int)      
    parser.add_argument('--lr0',default=0.1,type=float)  
    parser.add_argument('--coupling',default=4,type=int) 
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--size',default=200,type=int)      
    parser.add_argument('--bs',default=32,type=int)  
    parser.add_argument('--epochs',default=40,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--model',default='BHN_MLPWN',type=str)
    parser.add_argument('--anneal',default=0,type=int)
    parser.add_argument('--n_hiddens',default=2,type=int)
    parser.add_argument('--n_units',default=100,type=int)
    parser.add_argument('--totrain',default=1,type=int)
    parser.add_argument('--seed',default=427,type=int)
    parser.add_argument('--override',default=1,type=int)
    parser.add_argument('--reinit',default=0,type=int)
    parser.add_argument('--acquisition_function',default='bald',type=str)
    parser.add_argument('--data_name',default='boston',type=str)
    parser.add_argument('--flow',default='IAF',type=str, choices=['RealNVP', 'IAF'])
    parser.add_argument('--save_dir',default='./models',type=str)
    parser.add_argument('--save_results',default='./al_results/',type=str)
    
    
    args = parser.parse_args()
    print args
    
    
    set_rng(np.random.RandomState(args.seed))
    np.random.seed(args.seed+1000)
    
    if args.prior == 'log_normal':
        pr = 0
    if args.prior == 'log_laplace':
        pr = 1    
    if args.model == 'BHN_MLPWN':
        md = 0
    if args.model == 'MCdropout_MLP':
        md = 1
    
    
    path = args.save_dir
    name = '{}/mnistWN_md{}nh{}nu{}c{}pr{}lbda{}lr0{}lrd{}an{}s{}seed{}reinit{}flow{}'.format(
        path,
        md,
        args.n_hiddens,
        args.n_units,
        args.coupling,
        pr,
        args.lbda,
        args.lr0,
        args.lrdecay,
        args.anneal,
        args.size,
        args.seed,
        args.reinit,
        args.flow
    )

    coupling = args.coupling
    perdatapoint = args.perdatapoint
    lrdecay = args.lrdecay
    lr0 = args.lr0
    lbda = np.cast['float32'](args.lbda)
    bs = args.bs
    epochs = args.epochs
    n_hiddens = args.n_hiddens
    n_units = args.n_units
    anneal = args.anneal

    dataset_name = args.data_name

    if args.prior=='log_normal':
        prior = log_normal
    elif args.prior=='log_laplace':
        prior = log_laplace
    else:
        raise Exception('no prior named `{}`'.format(args.prior))
    size = max(10,min(50000,args.size))
    
    if os.path.isfile('/data/lisa/data/mnist.pkl.gz'):
        filename = '/data/lisa/data/mnist.pkl.gz'
    elif os.path.isfile(r'./data/mnist.pkl.gz'):
        filename = r'./data/mnist.pkl.gz'
    else:        
        print '\n\tdownloading mnist'
        import download_datasets.mnist
        filename = r'./data/mnist.pkl.gz'


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
    	raise Exception('dataset too big!!! TO DO')

    else:
    	raise Exception('Need a valid dataset')

    train_x, train_y, valid_x, valid_y, test_x , test_y = get_dataset(data)

    train_y = train_y.reshape((train_y.shape[0],1))
    valid_y = valid_y.reshape((valid_y.shape[0], 1))
    test_y = test_y.reshape((test_y.shape[0],1))

    input_dim = train_x.shape[1]

    if args.reinit:
        init_batch_size = 16
        init_batch = train_x[:size][-init_batch_size:]


    else:
        init_batch = None
        
    if args.model == 'BHN_MLPWN':
        model = MLPWeightNorm_BHN(lbda=lbda,
                                  perdatapoint=perdatapoint,
                                  srng = RandomStreams(seed=args.seed+2000),
                                  prior=prior,
                                  coupling=coupling,
                                  n_hiddens=n_hiddens,
                                  n_units=n_units,
                                  input_dim=input_dim,
                                  flow=args.flow,
                                  init_batch=init_batch)
    elif args.model == 'MCdropout_MLP':
        model = MCdropout_MLP(n_hiddens=n_hiddens,
                              n_units=n_units)
    else:
        raise Exception('no model named `{}`'.format(args.model))

    va_rec_name = name+'_recs'
    tr_rec_name = name+'_recs_train' # TODO (we're already saving the valid_recs!)
    save_path = name + '.params.npy'
    if os.path.isfile(save_path) and not args.override:
        print 'load best model'
        e0 = model.load(save_path)
        va_recs = open(va_rec_name,'r').read().split('\n')[:e0]
        #tr_recs = open(tr_rec_name,'r').read().split('\n')[:e0]
        rec = max([float(r) for r in va_recs])
        
    else:
        e0 = 0
        rec = 0



    if args.acquisition_function == 'bald':
        active_learning_acquisition_function = active_learning_acquisition_bald
    elif args.acquisition_function == 'highest_entropy':
        active_learning_acquisition_function = active_learning_acquisition_highest_entropy

    elif args.acquisition_function == 'highest_variance':
        active_learning_acquisition_function = active_learning_acquisition_highest_variance
        
    else:
        raise Exception('Select correct acquisition function for active learning')


    #for storing results
    acquisition_iterations = 9
    all_test_rmse_per_acq = 0
    all_train_rmse_per_acq = 0

    #split dataset into train, valid, pool set and test set
    pool_x, pool_y, train_x, train_y, valid_x, valid_y, test_x, test_y = get_active_learning_dataset_split(train_x, train_y, valid_x, valid_y, test_x, test_y)

    #train the model with given training data
    training_rmse, validation_rmse, training_rmse_with_current_data, validation_rmse_with_current_data = train_model(model.train_func,model.predict,
            train_x[:size],train_y[:size],
            valid_x,valid_y,
            lr0,lrdecay,bs,epochs,anneal,name,
            e0,rec)


    #evaluate the model for current training data
    te_rmse = evaluate_model(model.predict_proba, test_x,test_y)


    all_train_rmse_per_acq = training_rmse_with_current_data
    all_valid_rmse_per_acq = validation_rmse_with_current_data
    all_test_rmse_per_acq = te_rmse


    print('Starting Active Learning Experiments')

    for i in range(acquisition_iterations):

        print ("Acquisition Iterations", i)

        #compute uncertainty over pool set points, return query points
        pool_x, pool_y, train_x, train_y = active_learning_acquisition_function(model.predict_proba, pool_x, pool_y, train_x, train_y)

        training_rmse, validation_rmse, training_rmse_with_current_data, validation_rmse_with_current_data = train_model(model.train_func,model.predict,
                train_x[:size],train_y[:size],
                valid_x,valid_y,
                lr0,lrdecay,bs,epochs,anneal,name,
                e0,rec)

        te_rmse = evaluate_model(model.predict_proba, test_x,test_y)

        all_test_rmse_per_acq = np.append(all_test_rmse_per_acq, te_rmse)
        all_train_rmse_per_acq = np.append(all_train_rmse_per_acq, training_rmse_with_current_data)
        all_valid_rmse_per_acq = np.append(all_valid_rmse_per_acq, validation_rmse_with_current_data)


    np.save(args.save_results + args.data_name + "_trainining_rmse.npy", all_train_rmse_per_acq)
    np.save(args.save_results + args.data_name + "_validation_rmse.npy", all_valid_rmse_per_acq)
    np.save(args.save_results + args.data_name + "_test_rmse.npy", all_test_rmse_per_acq)






