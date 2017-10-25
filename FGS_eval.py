# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:11:43 2017

@author: Chin-Wei
"""



import numpy as np
#from cleverhans import attacks_th
import theano
import theano.tensor as T

from sklearn.metrics import roc_auc_score
from acquisition_functions import bald, max_ent, var_ratio, mean_std
score_fs = [bald, max_ent, var_ratio, mean_std]
eval_scores = lambda sps: [f(sps) for f in score_fs]

                           
                           
                           
def fgm_grad(x, predictions, y):
    loss = T.nnet.categorical_crossentropy(predictions,y)    
    grad = T.grad(loss.mean(), x)
    return grad

    
def evaluate(X,Y,predict_proba,
             input_var,target_var,prediction,
             eps=[0.001,0.002,0.003,0.004,0.005,0.008,0.01,0.012,0.015,
                  0.02,0.025,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.3,0.5],
             max_n=100,n_mc=20,n_classes=10,
             avg = 10):
    
    print 'compiling attacker ...'
    
    #attack = attacks_th.fgm(input_var,prediction,target_var,1.0) - input_var
    grad_ = fgm_grad(input_var,prediction,target_var)
    grad = theano.function([input_var,target_var],grad_)
    def att(x,y,ep):
        grads = sum([grad(x,y) for i in range(avg)]) / float(avg)
        signed = np.sign(grads)
        return x + ep * signed
        
    #att_ = theano.function([input_var,target_var],attack)
    #att = lambda x,y,ep: x + ep * \
    #                         sum([att_(x,y) for i in range(avg)]) / float(avg)
    
    N = X.shape[0]
    num_batches = np.ceil(N / float(max_n)).astype(int)
    
    def per_ep(ep):
        Xa = np.zeros(X.shape,dtype='float32')
        for j in range(num_batches):
            x = X[j*max_n:(j+1)*max_n]
            y = Y[j*max_n:(j+1)*max_n]
            xa = att(x,y,ep)
            Xa[j*max_n:(j+1)*max_n] = xa
        
        MCt = np.zeros((n_mc,X.shape[0],n_classes),dtype='float32')    
        for i in range(n_mc):
            for j in range(num_batches):
                x = Xa[j*max_n:(j+1)*max_n]
                MCt[i,j*max_n:(j+1)*max_n] = predict_proba(x)
        
        Y_proba = MCt.mean(0)
        
        # generalization
        Y_pred = Y_proba.argmax(-1)
        Y_true = Y.argmax(-1)
        corr = np.equal(Y_pred,Y_true)
        err = 1-corr
        acc = corr.mean()
        
        #score function
        #Y_entropy = np.sum(Y_proba * np.log(Y_proba),1)
        #Y_max = Y_proba.max(1)
        #Y_mstd = MCt.std(0).mean(1)
        Y_bald, Y_entropy, Y_max, Y_mstd = eval_scores(MCt)
        return acc, Y_bald, Y_entropy, Y_max, Y_mstd, err
    
    accs = list()
    ood_blds = list()
    ood_ents = list()
    ood_maxs = list()
    ood_stds = list()
    erd_blds = list()
    erd_ents = list()
    erd_maxs = list()
    erd_stds = list()
    blds = list()
    ents = list()
    maxs = list()
    stds = list()
    
    acc0, Y_bald0, Y_entropy0, Y_max0, Y_mstd0, err0 = per_ep(0.0)
    accs.append(acc0)
    blds.append(Y_bald0.mean())
    ents.append(Y_entropy0.mean())
    maxs.append(Y_max0.mean())
    stds.append(Y_mstd0.mean())
    
    erd_blds.append(roc_auc_score(err0,Y_bald0))
    erd_ents.append(roc_auc_score(err0,Y_entropy0))
    erd_maxs.append(roc_auc_score(err0,Y_max0))
    erd_stds.append(roc_auc_score(err0,Y_mstd0))
    
    for ep in eps:
        acc, Y_bald, Y_entropy, Y_max, Y_mstd, err = per_ep(ep)
        accs.append(acc.mean())
        blds.append(Y_bald.mean())
        ents.append(Y_entropy.mean())
        maxs.append(Y_max.mean())
        stds.append(Y_mstd.mean())
        
        # entropy
        predn = np.concatenate([Y_bald0,Y_bald])
        truth = np.ones((Y_bald0.shape[0]*2))
        truth[:Y_bald0.shape[0]] = 0
        sc_bld = roc_auc_score(truth,predn)
        
        # entropy
        predn = np.concatenate([Y_entropy0,Y_entropy])
        truth = np.ones((Y_entropy0.shape[0]*2))
        truth[:Y_entropy0.shape[0]] = 0
        sc_ent = roc_auc_score(truth,predn)
        
        # max softmax
        predn = np.concatenate([Y_max0,Y_max])
        truth = np.ones((Y_entropy0.shape[0]*2))
        truth[:Y_entropy0.shape[0]] = 0
        sc_max = roc_auc_score(truth,predn)
        
        # entropy
        predn = np.concatenate([Y_mstd0,Y_mstd])
        truth = np.ones((Y_entropy0.shape[0]*2))
        truth[:Y_entropy0.shape[0]] = 0
        sc_std = roc_auc_score(truth,predn)
        
        print ep, sc_bld, sc_ent, sc_max, sc_std
        ood_blds.append(sc_bld)
        ood_ents.append(sc_ent)
        ood_maxs.append(sc_max)
        ood_stds.append(sc_std)
        erd_blds.append(roc_auc_score(err0,Y_bald))
        erd_ents.append(roc_auc_score(err,Y_entropy))
        erd_maxs.append(roc_auc_score(err,Y_max))
        erd_stds.append(roc_auc_score(err,Y_mstd))
        
    return accs, blds, ents, maxs, stds, \
           ood_blds, ood_ents, ood_maxs, ood_stds, \
           erd_blds, erd_ents, erd_maxs, erd_stds

    
        
        
        
        
        
