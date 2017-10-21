#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: Riashat Islam
"""

import math
import numpy
np = numpy
np.random.seed(1) # TODO
import os
import random
import pandas as pd

from BHNs_MLP_Regression import MLPWeightNorm_BHN, MCdropout_MLP, Backprop_MLP
from get_regression_data import get_regression_dataset
from ops import load_mnist
from utils import log_normal, log_laplace

import lasagne
import theano
import theano.tensor as T
from lasagne.random import set_rng
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt

lrdefault = 1e-3    
    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def train_model(train_func,predict_func,X,Y,Xv,Yv,
                lr0=0.1,lrdecay=1,bs=32,epochs=50,anneal=0,name='0',
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


        tr_rmse = evaluate_model(model.predict, X, Y, n_mc=50)   
        print '\n\ntr rmse at epochs {}: {}'.format(e,tr_rmse)    
        va_rmse = evaluate_model(model.predict,Xv,Yv,n_mc=50)
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


    return training_rmse, validation_rmse
    


def evaluate_model(predict,X,Y,n_mc=100,max_n=100):
    MCt = np.zeros((n_mc,X.shape[0],1))
    
    N = X.shape[0]
    num_batches = np.ceil(N / float(max_n)).astype(int)
    for i in range(n_mc):
        for j in range(num_batches):
            x = X[j*max_n:(j+1)*max_n]
            MCt[i,j*max_n:(j+1)*max_n] = predict(x)

    Y_pred = MCt.mean(0)
    Y_true = Y
    RMSE = rmse(Y_pred, Y_true)

    return RMSE




def plot_rmse(stats1, stats2, save_results, dataset_name, smoothing_window=1, noshow=False):

    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(22)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'28'}

    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(rewards_smoothed_1, color = "#1f77b4", linewidth=2.5, label="Training RMSE")    
    cum_rwd_2, = plt.plot(rewards_smoothed_2, color = "#ff7f0e", linewidth=2.5, label="Validation RMSE" )  


    plt.legend(handles=[cum_rwd_1, cum_rwd_2],  loc='lower right', prop={'size' : 16})
    plt.xlabel("Number of Epochs",**axis_font)
    plt.ylabel("Accuracy", **axis_font)
    plt.title("Training and Validation RMSE", **axis_font)
    plt.show()

    fig.savefig(save_results + dataset_name + '_train_valid_rmse.png')
    
    return fig







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
    parser.add_argument('--epochs',default=1000,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--model',default='BHN',type=str, choices=['BHN', 'MCDropout', 'Backprop'])
    parser.add_argument('--anneal',default=0,type=int)
    parser.add_argument('--n_hiddens',default=1,type=int)
    parser.add_argument('--n_units',default=50,type=int)
    parser.add_argument('--totrain',default=1,type=int)
    parser.add_argument('--seed',default=427,type=int)
    parser.add_argument('--override',default=1,type=int)
    parser.add_argument('--reinit',default=0,type=int)
    parser.add_argument('--dataset_name',default='boston',type=str, choices=['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'year'])
    parser.add_argument('--flow',default='IAF',type=str, choices=['RealNVP', 'IAF'])
    parser.add_argument('--save_dir',default=None, type=str)
    #parser.add_argument('--save_results',default='./results/',type=str)
    
    
    args = parser.parse_args()
    print args
    
    
    set_rng(np.random.RandomState(args.seed))
    np.random.seed(args.seed+1000)
    
    if args.prior == 'log_normal':
        pr = 0
    if args.prior == 'log_laplace':
        pr = 1    
    if args.model == 'BHN':
        md = 0
    if args.model == 'MCDropout':
        md = 1
    if args.model == 'Backprop':
        md = 2
    
    path = args.save_dir
    name = '{}/regressionWN_md{}nh{}nu{}c{}pr{}lbda{}lr0{}lrd{}an{}s{}seed{}reinit{}flow{}'.format(
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

    dataset_name = args.dataset_name

    if args.prior=='log_normal':
        prior = log_normal
    elif args.prior=='log_laplace':
        prior = log_laplace
    else:
        raise Exception('no prior named `{}`'.format(args.prior))
    size = max(10,min(50000,args.size))
    
    input_dim, train_x, train_y, valid_x, valid_y, test_x , test_y = get_regression_dataset(dataset_name)

    if args.reinit:
        init_batch_size = 64
        init_batch = train_x[:size][-init_batch_size:]

    else:
        init_batch = None
        
    if args.model == 'BHN':
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
    elif args.model == 'MCDropout':
        model = MCdropout_MLP(n_hiddens=n_hiddens,
                              n_units=n_units,
                              input_dim=input_dim)


    elif args.model =='Backprop':
        model = Backprop_MLP(n_hiddens=n_hiddens,
                              n_units=n_units,
                              input_dim=input_dim)

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

    if args.totrain:
        print '\nstart training from epoch {}'.format(e0)
        all_trainining_rmse, all_validation_rmse = train_model(model.train_func,model.predict,
                    train_x[:size],train_y[:size],
                    valid_x,valid_y,
                    lr0,lrdecay,bs,epochs,anneal,name,
                    e0,rec)
    else:
        print '\nno training'
    

    tr_rmse = evaluate_model(model.predict_proba,
                            train_x[:size],train_y[:size])
    print 'train rmse: {}'.format(tr_rmse)
                   
    va_rmse = evaluate_model(model.predict_proba,
                            valid_x,valid_y)
    print 'valid acc: {}'.format(va_rmse)
    
    ##### for test accuracy
    # te_rmse = evaluate_model(model.predict_proba,
    #                         test_x,test_y)
    # print 'test acc: {}'.format(te_rmse)
    #
    # print ("TEST ACCURACY", te_rmse)
    # for plotting
    # plot_rmse(all_trainining_rmse, all_validation_rmse, args.save_results, args.dataset_name)
	

    if save_dir is not None:
        np.save(name + "_all_training_rmse.npy", all_trainining_rmse)
        np.save(name + "_all_validation_rmse.npy", all_validation_rmse)
        np.save(name + "_training_rmse.npy", tr_rmse)
        np.save(name + "_validation_rmse.npy", va_rmse)








