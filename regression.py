#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: David Krueger
"""

import time
t0 = time.time()
times = {}
#times['start'] = t0

import argparse
import os
import sys
import time
import numpy 
np = numpy
#np.random.seed(1) # TODO

import math
import os
import random
import pandas as pd

from BHNs_MLP_Regression import MLPWeightNorm_BHN, MCdropout_MLP, Backprop_MLP
from dk_get_regression_data import get_regression_dataset
#from ops import load_mnist
from utils import log_normal, log_laplace

import lasagne
import theano
import theano.tensor as T
from lasagne.random import set_rng
from theano.tensor.shared_randomstreams import RandomStreams
#import matplotlib.pyplot as plt

import scipy

from logsumexp import logsumexp

#import shutil  # for eval_only


t1 = time.time()
times['imports'] = t1 - t0


# TODO:
n_mc = 20


"""
The idea here is that we will spit out all of the jobs on all of the datasets, and aggregate performance manually.
We will use grid search for now.
"""

def save_list(path, ll):
    thefile = open(path, 'w')
    for item in ll:
        thefile.write("%s\n" % item)
    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def get_lbda(tau, length_scale, drop_prob=None):
    # this is eqn (7) from https://arxiv.org/pdf/1506.02142.pdf (Gal)
    lbda = length_scale**2 / tau # we don't divide by the 2 * N(=dataset size) as in Gal, since our prior handles this scaling
    if drop_prob is not None:
        lbda *= (1-drop_prob)
    lbda = np.cast['float32'](lbda)
    return lbda

def get_LL(y_hat, y, tau):
    # this is eqn (8) from https://arxiv.org/pdf/1506.02142.pdf (Gal)
    n_mc = len(y_hat)
    #print "get_LL... n_mc=", n_mc
    return (logsumexp(-.5*tau*(y_hat-y)**2, axis=0) - np.log(n_mc) - .5*np.log(2*np.pi) - .5*np.log(tau**-1)).mean()


def train_model(model, save_,save_path,
                X,Y,Xv,Yv,
                Xt, Yt, # TODO: default to None
                y_mean, y_std,
                lr0,lrdecay,bs,epochs,anneal,
                e0=0, rec=0, taus=None,
                timing=True):
                #save_=True):
    
    if timing:
        start_time = time.time()
        eval_time = 0

    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    

    tr_RMSEs = list()
    va_RMSEs = list()
    te_RMSEs = list()
    tr_LLs = [[] for tau in taus]
    va_LLs = [[] for tau in taus]
    te_LLs = [[] for tau in taus]
    best_va_LLs = [-np.inf,] * len(taus)
    
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
            loss = model.train_func(x,y,N,lr,w)

         
        if timing:
            eval_start = time.time()
        tr_rmse, tr_LL = evaluate_model(model.predict, X, Y, n_mc=n_mc, taus=taus)  
        va_rmse, va_LL = evaluate_model(model.predict,Xv,Yv, n_mc=n_mc, taus=taus, y_mean=y_mean, y_std=y_std)
        te_rmse, te_LL = evaluate_model(model.predict,Xt,Yt, n_mc=n_mc, taus=taus, y_mean=y_mean, y_std=y_std)
        if timing:
            eval_time += time.time() - eval_start

        if e % 5 == 0:
            if 0: #verbose
                print '\n'
                print 'tr LL at epochs {}: {}'.format(e,tr_LL)    
                print 'tr rmse at epochs {}: {}'.format(e,tr_rmse)    
            print 'va LL at epochs {}: {}'.format(e,va_LL)    
            #print 'va rmse at epochs {}: {}'.format(e,va_rmse)    
        
        tr_RMSEs.append(tr_rmse)
        va_RMSEs.append(va_rmse)
        te_RMSEs.append(te_rmse)
        # 
        [tr_LLs[n].append(tr_LL[n]) for n, _ in enumerate(taus)]
        [va_LLs[n].append(va_LL[n]) for n, _ in enumerate(taus)]
        [te_LLs[n].append(te_LL[n]) for n, _ in enumerate(taus)]
        
        for n, tau in enumerate(taus):
            if va_LL[n] > best_va_LLs[n]:
                best_va_LLs[n] = va_LL[n]
                if save_:
                    model.save(save_path + '_best_tau=' + str(tau))


    if timing:
        train_time = time.time() - start_time - eval_time
        return tr_LLs, tr_RMSEs, va_LLs, va_RMSEs, te_LLs, te_RMSEs, train_time, eval_time
    else:
        return tr_LLs, tr_RMSEs, va_LLs, va_RMSEs, te_LLs, te_RMSEs
    


def evaluate_model(predict,X,Y,
        y_mean=None, y_std=None,
        n_mc=100,max_n=100, taus=10.**np.arange(-3,6)):

    MCt = np.zeros((n_mc,X.shape[0],1))
    N = X.shape[0]
    num_batches = np.ceil(N / float(max_n)).astype(int)
    for i in range(n_mc):
        for j in range(num_batches):
            x = X[j*max_n:(j+1)*max_n]
            MCt[i,j*max_n:(j+1)*max_n] = predict(x)

    if y_std is not None:
        MCt *= y_std
    if y_mean is not None:
        MCt += y_mean

    LLs = [ get_LL(MCt, Y, tau) for tau in taus]
    y_hat = MCt.mean(0)
    RMSE = rmse(y_hat, Y)
    return RMSE, LLs





if __name__ == '__main__':
    

    taus = 10.**np.arange(-3,6)

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lrdecay',default=0.0,type=int)      
    parser.add_argument('--lr0',default=0.001,type=float)
    parser.add_argument('--coupling',default=4,type=int) 
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--bs',default=32,type=int)  
    parser.add_argument('--epochs',default=1000,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--model',default='BHN',type=str, choices=['BHN', 'MCD', 'Backprop'])
    parser.add_argument('--anneal',default=0,type=int)
    parser.add_argument('--n_hiddens',default=1,type=int)
    parser.add_argument('--n_units',default=50,type=int)
    parser.add_argument('--seed',default=None,type=int)
    parser.add_argument('--override',default=1,type=int)
    parser.add_argument('--reinit',default=1,type=int)
    # 
    parser.add_argument('--dataset',default='airfoil',type=str, choices=['airfoil', 'parkinsons'] + ['boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'year'])
    parser.add_argument('--train_on_valid',default=0, type=int, help="whether to train on the validation set")
    parser.add_argument('--data_path',default=None, type=str)
    parser.add_argument('--flow',default='IAF',type=str, choices=['RealNVP', 'IAF'])
    parser.add_argument('--save_dir',default=None, type=str)
    #
    parser.add_argument('--drop_prob',default=0.005, type=float) # .05, .01, .005
    #
    #parser.add_argument('--length_scale',default=1e-3, type=float) # 1e0,-1,-2,-4
    #parser.add_argument('--tau',default=1e2, type=float) # search over 
    #parser.add_argument('--normalization',default='by_train_set', type=str)
    parser.add_argument('--fname',default=None, type=str) # override this for launching with SLURM!!!
    parser.add_argument('--split',default=0, type=str) # TODO: , help="defaults to None, in which case this script will launch a copy of itself on ALL of the available splits")
    parser.add_argument('--eval_only',default=0, type=int, help="just run the final evaluation, NO training!")
    #parser.add_argument('--analyze',default=0, type=int, help="just run the final evaluation, NO training!")

    #parser.add_argument('--save_results',default='./results/',type=str)
    
    
    args = parser.parse_args()
    print args
    args_dict = args.__dict__
    # moved from below!
    locals().update(args_dict)

    eval_only = int(args_dict.pop('eval_only'))

    flags = [flag.lstrip('--') for flag in sys.argv[1:] if (not flag.startswith('--save_dir') and not flag.startswith('--eval_only'))]
    exp_description = '_'.join(flags)


    fname = args_dict.pop('fname')
    if fname is None:
        fname = os.path.basename(__file__)

    # prepare to save model
    if args_dict['save_dir'] is None:
        save_ = False
        save_path = None
        print "\n\n\n\t\t\t\t WARNING: save_dir is None! Results will not be saved! \n\n\n"
    else:
        save_ = True
        # save_dir = filename + PROVIDED parser arguments
        save_dir = os.path.join(args_dict.pop('save_dir'), fname)
        save_path = save_dir + '___' + '_'.join(flags)
        print"\t\t save_dir=",  save_dir
        # make directory for results
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # save ALL parser arguments
        with open (save_path + '__exp_settings.txt', 'w') as f:
            for key in sorted(args_dict):
                f.write(key+'\t'+str(args_dict[key])+'\n')

    #assert lbda is None # we'
    
    if seed is None:
        seed = np.random.randint(2**32 - 1)
    set_rng(np.random.RandomState(seed))
    np.random.seed(seed+1000)

    if data_path is None:
        data_path = os.path.join(os.environ['HOME'], 'BayesianHypernetCW/')
    
    t2 = time.time()
    times['setup'] = t2 - t1

    # 
    if 1:
        input_dim, tr_x, tr_y, va_x, va_y, te_x, te_y, y_mean, y_std = get_regression_dataset(dataset, split, data_path=data_path)
        if train_on_valid:
            print tr_x.shape
            tr_x = np.concatenate((tr_x, va_x), axis=0)
            print tr_x.shape
            tr_y = np.concatenate((tr_y, va_y), axis=0)
        if model == 'MCD':
            #lbda = get_lbda(tau, length_scale, drop_prob)
            #print drop_prob, lbda
            network = MCdropout_MLP(n_hiddens=n_hiddens,
                                      n_units=n_units,
                                      lbda=lbda,
                                      input_dim=input_dim)
        elif model == 'BHN':
            #lbda = get_lbda(tau, length_scale)
            prior = log_normal
            #print lbda
            if reinit:
                init_batch_size = 64
                init_batch = tr_x[-init_batch_size:]
            else:
                init_batch = None

            network = MLPWeightNorm_BHN(lbda=lbda,
                                          srng = RandomStreams(seed=seed+2000),
                                          prior=prior,
                                          coupling=coupling,
                                          n_hiddens=n_hiddens,
                                          n_units=n_units,
                                          input_dim=input_dim,
                                          flow=flow,
                                          init_batch=init_batch)

        t3 = time.time()
        times['compile'] = t3 - t2

        if eval_only:
            network.load(save_path + '_final.npy')
            print "skipping training (eval_only=1)"
            t4 = time.time()

        else:
            print "begin training"
            result = train_model(network, save_, save_path,
                            tr_x,tr_y,
                            va_x,va_y,
                                                            te_x, te_y,
                                                            y_mean, y_std,
                            lr0,lrdecay,bs,epochs,anneal,
                            taus=taus)
            tr_LLs, tr_RMSEs, va_LLs, va_RMSEs, te_LLs, te_RMSEs, train_time, eval_time = result

            t4 = time.time()
            times['train'] = train_time
            times['eval'] = eval_time

        print "done training, begin final evaluation"
        #tr_RMSE, tr_LL = evaluate_model(network.predict, tr_x, tr_y, n_mc=10000, taus=taus)  
        va_RMSE, va_LL = evaluate_model(network.predict, va_x, va_y, n_mc=1000, taus=taus, y_mean=y_mean, y_std=y_std) 
        te_RMSE, te_LL = evaluate_model(network.predict, te_x, te_y, n_mc=1000, taus=taus, y_mean=y_mean, y_std=y_std) 
        #total_runtime = time.time() - t0 
        #print "total_runtime=", total_runtime 

        t5 = time.time()
        times['final_eval'] = t5 - t4
        
        if save_:
            if eval_only: 
                # we only overwrite the _FINAL_ results
                np.savetxt(save_path + "__eval_only__FINAL_va_RMSE=" + str(np.round(va_RMSE, 3)), [va_RMSE])
                np.savetxt(save_path + "__eval_only__FINAL_te_RMSE=" + str(np.round(te_RMSE, 3)), [te_RMSE])
                for n, tau in enumerate(taus):
                    np.savetxt(save_path + '_tau=' + str(tau) + "__eval_only__FINAL_va_LL=" + str(np.round(va_LL[n], 3)), [va_LL[n]])
                    np.savetxt(save_path + '_tau=' + str(tau) + "__eval_only__FINAL_te_LL=" + str(np.round(te_LL[n], 3)), [te_LL[n]])


            else:
                # final params
                network.save(save_path + '_final')

                # learning curves
                save_list(save_path + "_tr_RMSEs", tr_RMSEs)
                save_list(save_path + "_va_RMSEs", va_RMSEs)
                save_list(save_path + "_te_RMSEs", te_RMSEs)
                for n, tau in enumerate(taus):
                    save_list(save_path + '_tau=' + str(tau) + "_tr_LLs", tr_LLs[n])
                    save_list(save_path + '_tau=' + str(tau) + "_va_LLs", va_LLs[n])
                    save_list(save_path + '_tau=' + str(tau) + "_te_LLs", te_LLs[n])

                # final results (with full evaluation) TODO: EARLY STOPPING!!!
                #np.savetxt(save_path + "_FINAL_tr_RMSE=" + str(np.round(tr_RMSE, 3)), [tr_RMSE])
                np.savetxt(save_path + "_FINAL_va_RMSE=" + str(np.round(va_RMSE, 3)), [va_RMSE])
                np.savetxt(save_path + "_FINAL_te_RMSE=" + str(np.round(te_RMSE, 3)), [te_RMSE])
                for n, tau in enumerate(taus):
                    #np.savetxt(save_path + '_tau=' + str(tau) + "_FINAL_tr_LL=" + str(np.round(tr_LL[n], 3)), [tr_LL[n]])
                    np.savetxt(save_path + '_tau=' + str(tau) + "_FINAL_va_LL=" + str(np.round(va_LL[n], 3)), [va_LL[n]])
                    np.savetxt(save_path + '_tau=' + str(tau) + "_FINAL_te_LL=" + str(np.round(te_LL[n], 3)), [te_LL[n]])

                t6 = time.time()
                times['saving'] = t6 - t5
                total_runtime = t6 - t0
            
                print "total_runtime=", total_runtime
                print "sum(times.values())", sum(times.values())
                for k,v in times.items():
                    print k, np.round(v, 4)

                # TODO: save times
                np.savetxt(save_path + "________________total_runtime=" + str(total_runtime), [total_runtime])

                def to_str(times_dict):
                    rval = '____timing____'
                    for k, v in times_dict.items():
                        rval += k + '_' + str(np.round(v, 3)) + '__'
                    return rval

                np.save(save_path + to_str(times), times)



