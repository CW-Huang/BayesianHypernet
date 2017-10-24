    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
"""@author: Riashat Islam
"""
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
from get_regression_data import get_regression_dataset
from ops import load_mnist
from utils import log_normal, log_laplace

import lasagne
import theano
import theano.tensor as T
from lasagne.random import set_rng
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt

import scipy

lrdefault = 1e-3    

n_mc = 20

def save_list(path, ll):
    thefile = open(path, 'w')
    for item in ll:
        thefile.write("%s\n" % item)

def get_LL(y_hat, y, tau):
    n_mc = len(y_hat)
    return scipy.misc.logsumexp(-.5*tau*(y_hat-y)**2) - np.log(n_mc) - .5*np.log(2*np.pi) - .5*np.log(tau**-1)

    
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def get_lbda(tau, length_scale, drop_prob=None):
    # this is eqn (7) from https://arxiv.org/pdf/1506.02142.pdf (Gal)
    lbda = length_scale**2 / tau # we don't divide by the 2 * N(=dataset size) as in Gal, since our prior handles this scaling
    if drop_prob is not None:
        lbda *= (1-drop_prob)
    lbda = np.cast['float32'](lbda)
    return lbda

def train_model(model, save_path, save_,
                X,Y,Xv,Yv,
                Xt, Yt, # TODO: default to None
                lr0=0.1,lrdecay=1,bs=32,epochs=50,anneal=0,name='0',
                e0=0,rec=0, tau=None):
                #save_=True):
    
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    

    tr_RMSEs = list()
    tr_LLs = list()
    va_RMSEs = list()
    va_LLs = list()
    te_RMSEs = list()
    te_LLs = list()
    
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

            loss = model.train_func(x,y,N,lr,w)
            #print ("Loss", loss)

            if 0:#t%100==0:
                #print 'epoch: {} {}, loss:{}'.format(e,t,loss)

                tr_rmse = rmse(model.predict(X), Y)
                va_rmse = rmse(model.predict(Xv), Yv)
                # print '\ttrain rmse: {}'.format(tr_rmse)
                # print '\tvalid rmse: {}'.format(va_rmse)
            t+=1


        tr_rmse, tr_LL = evaluate_model(model.predict, X, Y, n_mc=n_mc, tau=tau)  
        va_rmse, va_LL = evaluate_model(model.predict,Xv,Yv, n_mc=n_mc, tau=tau)
        te_rmse, te_LL = evaluate_model(model.predict,Xt,Yt, n_mc=n_mc, tau=tau)
        if e % 5 == 0:
            if 0: #verbose
                print '\n'
                print 'tr LL at epochs {}: {}'.format(e,tr_LL)    
                print 'tr rmse at epochs {}: {}'.format(e,tr_rmse)    
            print 'va LL at epochs {}: {}'.format(e,va_LL)    
            #print 'va rmse at epochs {}: {}'.format(e,va_rmse)    
        
        tr_RMSEs.append(tr_rmse)
        tr_LLs.append(tr_LL)
        va_RMSEs.append(va_rmse)
        va_LLs.append(va_LL)
        te_RMSEs.append(te_rmse)
        te_LLs.append(te_LL)
        
        if va_LL > rec and save_:
            print '.... save best model .... '
            model.save(save_path,[e])
            rec = va_LL
        #print '\n\n'

    return tr_LLs, tr_RMSEs, va_LLs, va_RMSEs, te_LLs, te_RMSEs
    


def evaluate_model(predict,X,Y,n_mc=100,max_n=100, tau=None):
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

    LL = get_LL(Y_pred, Y_true, tau)

    return RMSE, LL





if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--lrdecay',default=0.0,type=int)      
    parser.add_argument('--lr0',default=0.001,type=float) # .1
    parser.add_argument('--coupling',default=4,type=int) 
    parser.add_argument('--lbda',default=None,type=float)  
    parser.add_argument('--size',default=2000,type=int)      
    parser.add_argument('--bs',default=32,type=int)  
    parser.add_argument('--epochs',default=1000,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--model',default='BHN',type=str, choices=['BHN', 'MCDropout', 'Backprop'])
    parser.add_argument('--anneal',default=0,type=int)
    parser.add_argument('--n_hiddens',default=1,type=int) # 1
    parser.add_argument('--n_trials',default=10,type=int)
    parser.add_argument('--n_units',default=50,type=int) # 50
    parser.add_argument('--totrain',default=1,type=int)
    parser.add_argument('--seed',default=None,type=int)
    parser.add_argument('--override',default=1,type=int)
    parser.add_argument('--reinit',default=1,type=int)
    parser.add_argument('--dataset',default='airfoil',type=str, choices=['airfoil', 'parkinsons',  'boston', 'concrete', 'energy', 'kin8nm', 'naval', 'power', 'protein', 'wine', 'yacht', 'year'])
    parser.add_argument('--flow',default='IAF',type=str, choices=['RealNVP', 'IAF'])
    parser.add_argument('--save_dir',default=None, type=str)
    parser.add_argument('--cross_validate',default=0, type=int)
    parser.add_argument('--drop_prob',default=0.005, type=float)
    parser.add_argument('--length_scale',default=1e-3, type=float)
    parser.add_argument('--tau',default=1e2, type=float)
    parser.add_argument('--grid_search',default=0, type=int)
    #parser.add_argument('--save_results',default='./results/',type=str)
    
    
    args = parser.parse_args()
    print args
    args_dict = args.__dict__
    flags = [flag.lstrip('--') for flag in sys.argv[1:] if not flag.startswith('--save_dir')]
    exp_description = '_'.join(flags)

    if args_dict['save_dir'] is None:
        save_ = False
        print "\n\n\n\t\t\t\t WARNING: save_dir is None! Results will not be saved! \n\n\n"
    else:
        save_ = True
        # save_dir = filename + PROVIDED parser arguments
        save_dir = os.path.join(args_dict.pop('save_dir'), os.path.basename(__file__) + '___' + '_'.join(flags))
        print("\t\t save_dir=",  save_dir)

        # make directory for results
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # save ALL parser arguments
        with open (os.path.join(save_dir,'exp_settings.txt'), 'w') as f:
            for key in sorted(args_dict):
                f.write(key+'\t'+str(args_dict[key])+'\n')

    locals().update(args_dict)

    assert lbda is None
    assert size == 2000
    
    if seed is None:
        seed = np.random.randint(2**32 - 1)
    set_rng(np.random.RandomState(seed))
    np.random.seed(seed+1000)
    
    input_dim, train_x, train_y, valid_x, valid_y, test_x , test_y = get_regression_dataset(dataset)
    datasets = [get_regression_dataset(dataset) for _ in range(n_trials)]

    # SET HPARAMS FOR SEARCH (override grid search if provided as flag)
    if grid_search:
        # TODO: better grid...
        length_scales = 10.**np.arange(-4,5)[::-1]#length_scales = [.1, .01,  .001] # length scale should be smaller!
        taus = 10.**np.arange(0,6)[::-1]#taus = [.01, .1, 1, 10., 100.] # tau should be larger!
        lr0s = [.01, .001, .0001]#lr0s = [.01, .001, .0001]
        drop_probs = [.005, 0.01, 0.1]#[.1, .05, .02, .01, .005, .002, .001]

        for trial, dataset in enumerate(datasets):

            input_dim, train_x, train_y, valid_x, valid_y, test_x , test_y = dataset

            for length_scale in length_scales:
                for tau in taus:
                    for lr0 in lr0s:
                        # DROPOUT
                        if model == 'MCDropout':
                            for drop_prob in drop_probs:
                                t0 = time.time()
                                lbda = get_lbda(tau, length_scale, drop_prob)
                                print drop_prob, lbda
                                network = MCdropout_MLP(n_hiddens=n_hiddens,
                                        n_units=n_units,
                                                          lbda=lbda,
                                                        #srng = RandomStreams(seed=seed+2000),
                                                          input_dim=input_dim)

                                path = save_dir
                                name = '{}/airfoil_regression_m{}p{}c{}lr0{}seed{}reinit{}flow{}trial{}tau{}l{}'.format(
                                    path,
                                    model,
                                    drop_prob,
                                    coupling,
                                    lr0,
                                    seed,
                                    reinit,
                                    flow,
                                    trial,
                                    tau,
                                    length_scale
                                )

                                e0 = 0
                                rec = 0
                                tr_LLs, tr_RMSEs, va_LLs, va_RMSEs, te_LLs, te_RMSEs = train_model(network, name, save_,
                                                train_x[:size],train_y[:size],
                                                valid_x,valid_y,
                                                        test_x, test_y,
                                                lr0,lrdecay,bs,epochs,anneal,name,
                                                e0,rec, tau)
                                
                                if save_:
                                    save_list(name + "_tr_RMSE", tr_RMSEs)
                                    save_list(name + "_tr_LL", tr_LLs)
                                    save_list(name + "_va_RMSE", va_RMSEs)
                                    save_list(name + "_va_LL", va_LLs)
                                    save_list(name + "_te_RMSE", te_RMSEs)
                                    save_list(name + "_te_LL", te_LLs)
                                print "time=", time.time() - t0



                        # BHN
                        elif model == 'BHN':
                            lbda = get_lbda(tau, length_scale)
                            prior = log_normal
                            for coupling in [4]:#,12]:
                                for flow in ['IAF']:#, 'RealNVP']:
                                    for reinit in [1]:#,0]:
                                        t0 = time.time()
                                        print lbda
                                        if reinit:
                                            init_batch_size = 64
                                            init_batch = train_x[:size][-init_batch_size:]
                                        else:
                                            init_batch = None

                                        network = MLPWeightNorm_BHN(lbda=lbda,
                                                                      perdatapoint=perdatapoint,
                                                                      srng = RandomStreams(seed=seed+2000),
                                                                      prior=prior,
                                                                      coupling=coupling,
                                                                      n_hiddens=n_hiddens,
                                                                      n_units=n_units,
                                                                      input_dim=input_dim,
                                                                      flow=flow,
                                                                      init_batch=init_batch)
                        
                                        path = save_dir
                                        name = '{}/airfoil_regression_m{}p{}c{}lr0{}seed{}reinit{}flow{}trial{}tau{}l{}'.format(
                                            path,
                                            model,
                                            drop_prob,
                                            coupling,
                                            lr0,
                                            seed,
                                            reinit,
                                            flow,
                                            trial,
                                            tau,
                                            length_scale
                                        )
                                    

                                        e0 = 0
                                        rec = 0
                                        tr_LLs, tr_RMSEs, va_LLs, va_RMSEs, te_LLs, te_RMSEs = train_model(network, name, save_,
                                                        train_x[:size],train_y[:size],
                                                        valid_x,valid_y,
                                                        test_x, test_y,
                                                        lr0,lrdecay,bs,epochs,anneal,name,
                                                        e0,rec, tau)
                                        
                                        if save_:
                                            save_list(name + "_tr_RMSE", tr_RMSEs)
                                            save_list(name + "_tr_LL", tr_LLs)
                                            save_list(name + "_va_RMSE", va_RMSEs)
                                            save_list(name + "_va_LL", va_LLs)
                                            save_list(name + "_te_RMSE", te_RMSEs)
                                            save_list(name + "_te_LL", te_LLs)
                                        print "time=", time.time() - t0

    else:
        input_dim, train_x, train_y, valid_x, valid_y, test_x , test_y = get_regression_dataset(dataset)
        trial = 0
        if model == 'MCDropout':
            t0 = time.time()
            lbda = get_lbda(tau, length_scale, drop_prob)
            print drop_prob, lbda
            network = MCdropout_MLP(n_hiddens=n_hiddens,
                    n_units=n_units,
                                      lbda=lbda,
                                                        #srng = RandomStreams(seed=seed+2000),
                                      input_dim=input_dim)
        elif model == 'BHN':
            lbda = get_lbda(tau, length_scale)
            prior = log_normal
            t0 = time.time()
            print lbda
            if reinit:
                init_batch_size = 64
                init_batch = train_x[:size][-init_batch_size:]
            else:
                init_batch = None

            network = MLPWeightNorm_BHN(lbda=lbda,
                                          perdatapoint=perdatapoint,
                                          srng = RandomStreams(seed=seed+2000),
                                          prior=prior,
                                          coupling=coupling,
                                          n_hiddens=n_hiddens,
                                          n_units=n_units,
                                          input_dim=input_dim,
                                          flow=flow,
                                          init_batch=init_batch)

        path = save_dir
        name = '{}/airfoil_regression_m{}p{}c{}lr0{}seed{}reinit{}flow{}trial{}tau{}l{}'.format(
            path,
            model,
            drop_prob,
            coupling,
            lr0,
            seed,
            reinit,
            flow,
            trial,
            tau,
            length_scale
        )

        e0 = 0
        rec = 0
        result = train_model(network, name, save_,
                        train_x[:size],train_y[:size],
                        valid_x,valid_y,
                                                        test_x, test_y,
                        lr0,lrdecay,bs,epochs,anneal,name,
                        e0,rec, tau)
        tr_LLs, tr_RMSEs, va_LLs, va_RMSEs, te_LLs, te_RMSEs = result
        
        if save_:
            save_list(name + "_tr_RMSE", tr_RMSEs)
            save_list(name + "_tr_LL", tr_LLs)
            save_list(name + "_va_RMSE", va_RMSEs)
            save_list(name + "_va_LL", va_LLs)
            save_list(name + "_te_RMSE", te_RMSEs)
            save_list(name + "_te_LL", te_LLs)
        print "time=", time.time() - t0


        # post-experiment analysis
        print "tr/va LL:", max(tr_LLs), max(va_LLs)
        print "tr/va RMSE:", max(tr_RMSEs), max(va_RMSEs)
        print "lambda", lbda
        # for these 2 lines to work, you need to run this script interactively in ipython with run -i SCRIPTNAME, and define flagz = []; results = [] in the interactive session
        #flagz = []; results = [] 
        # flagz.append(flags)
        # results.append(result)

        # desc= exp_description + '   lbda=' + str(lbda)

        # figure(1)
        # subplot(121)
        # plot(tr_LLs, label=desc+'_TR')
        # plot(va_LLs, label=desc)
        # subplot(122)
        # plot(tr_RMSEs, label=desc+'_TR')
        # plot(va_RMSEs, label=desc)
        # legend()

        # figure()
        # suptitle(desc)
        # subplot(121)
        # plot(tr_LLs)
        # plot(va_LLs)
        # subplot(122)
        # plot(tr_RMSEs)
        # plot(va_RMSEs)

        # show()
