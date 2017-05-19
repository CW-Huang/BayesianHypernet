#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:46:38 2017

@author: Chin-Wei
"""

from modules import LinearFlowLayer, IndexLayer, PermuteLayer, ReverseLayer
from modules import CoupledDenseLayer, stochasticDenseLayer2, ConvexBiasLayer
from modules import * # just in case
from utils import log_normal, log_stdnormal
from ops import load_mnist
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=427)
floatX = theano.config.floatX

import lasagne
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers import get_output
from lasagne.objectives import categorical_crossentropy as cc
import numpy as np


from helpers import flatten_list, gelu, plot_dict


# TODO: 
#   AD in script (debug)
#       look into numerical precision issues!!
#   init

NUM_CLASSES = 10


if 1:#def main():
    """
    MNIST example
    weight norm reparameterized MLP with prior on rescaling parameters
    """
    
    import argparse
    import sys
    import os
    import numpy 
    np = numpy
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',type=str, default='Dan', choices=['CW', 'Dan'])
    parser.add_argument('--bs',default=128,type=int)  
    parser.add_argument('--convex_combination', type=int, default=0) 
    parser.add_argument('--coupling', type=int, default=4) 
    parser.add_argument('--epochs', type=int, default=100)  
    parser.add_argument('--init', type=str, default='normal')  
    parser.add_argument('--lrdecay',action='store_true')  
    parser.add_argument('--lr0',default=0.001,type=float)  
    parser.add_argument('--lbda',default=1.,type=float)  
    parser.add_argument('--model', default='mlp', type=str, choices=['mlp', 'hnet', 'hnet2', 'dropout', 'dropout2', 'weight_uncertainty'])
    parser.add_argument('--nonlinearity',default='rectify', type=str)
    parser.add_argument('--perdatapoint',action='store_true')    
    parser.add_argument('--num_examples',default=1000,type=int)  
    parser.add_argument('--num_samples',default=10,type=int)  
    parser.add_argument('--size',default=50000,type=int)  
    parser.add_argument('--test_eval',default=0,type=int)  
    #
    #parser.add_argument('--save_path',default=None,type=str)  
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--verbose', type=int, default=1)


    # --------------------------------------------
    # PARSE ARGS and SET-UP SAVING and RANDOM SEED

    args = parser.parse_args()
    args_dict = args.__dict__

    # save_path = filename + PROVIDED parser arguments
    flags = [flag.lstrip('--') for flag in sys.argv[1:]]
    flags = [ff for ff in flags if not ff.startswith('save_dir')]
    save_dir = args_dict.pop('save_dir')
    save_path = os.path.join(save_dir, os.path.basename(__file__) + '___' + '_'.join(flags))
    args_dict['save_path'] = save_path

    if args_dict['save']:
        # make directory for results, save ALL parser arguments
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open (os.path.join(save_path,'exp_settings.txt'), 'w') as f:
            for key in sorted(args_dict):
                f.write(key+'\t'+str(args_dict[key])+'\n')
        print( save_path)
        #assert False

    locals().update(args_dict)

    if nonlinearity == 'rectify':
        nonlinearity = lasagne.nonlinearities.rectify
    elif nonlinearity == 'gelu':
        nonlinearity = gelu
    
    lbda = np.cast[floatX](args.lbda)
    size = max(10,min(50000,args.size))
    clip_grad = 100
    max_norm = 100

    # SET RANDOM SEED (TODO: rng vs. random.seed)
    if seed is not None:
        np.random.seed(seed)  # for reproducibility
        rng = numpy.random.RandomState(seed)
    else:
        rng = numpy.random.RandomState(np.random.randint(2**32 - 1))
    # --------------------------------------------


    # load dataset
    filename = '/data/lisa/data/mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
    
    
    input_var = T.matrix('input_var')
    target_var = T.matrix('target_var')
    dataset_size = T.scalar('dataset_size')
    lr = T.scalar('lr') 
    
    # 784 -> 20 -> 10
    if arch == 'CW':
        weight_shapes = [(784, 200), (200,  10)]
    elif arch == 'Dan':
        if model in ['hnet2', 'dropout2']:
            weight_shapes = [(784, 512), (512, 512), (512,512), (512,  10)]
        else:
            weight_shapes = [(784, 256), (256, 256), (256,256), (256,  10)]
    
    if model == 'weight_uncertainty': 
        num_params = sum(np.prod(ws) for ws in weight_shapes)
    else:
        num_params = sum(ws[1] for ws in weight_shapes)

    if perdatapoint:
        wd1 = input_var.shape[0]
    else:
        wd1 = 1

    if model in ['hnet', 'hnet2', 'weight_uncertainty']:
        # stochastic hypernet    
        ep = srng.normal(std=0.01,size=(wd1,num_params),dtype=floatX)
        logdets_layers = []
        h_layer = lasagne.layers.InputLayer([None,num_params])
        
        layer_temp = LinearFlowLayer(h_layer)
        h_layer = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))

        for c in range(coupling):
            h_layer = ReverseLayer(h_layer,num_params)
            layer_temp = CoupledDenseLayer(h_layer,10)
            h_layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))
        
        if convex_combination:
            if init == 'normal':
                h_layer = ConvexBiasLayer(h_layer, b=init.Normal(0.01, 0))
            else: # TODO
                assert False
                h_layer = ConvexBiasLayer(h_layer, b=init.Normal(0.01, 0))
            h_layer = IndexLayer(layer_temp,0)
            logdets_layers.append(IndexLayer(layer_temp,1))


        weights = lasagne.layers.get_output(h_layer,ep)
        
        # primary net
        t = np.cast['int32'](0)
        layer = lasagne.layers.InputLayer([None,784])
        inputs = {layer:input_var}
        for ws in weight_shapes:
            if model == 'weight_uncertainty':
                num_param = np.prod(ws)
            else:
                num_param = ws[1]
            w_layer = lasagne.layers.InputLayer((None,num_param))
            # TODO: why is reshape needed??
            weight = weights[:,t:t+num_param].reshape((wd1,num_param))
            inputs[w_layer] = weight
            layer = stochasticDenseLayer2([layer,w_layer],num_param, nonlinearity=nonlinearity)
            print layer.output_shape
            t += num_param

            
        layer.nonlinearity = nonlinearities.softmax
        y = get_output(layer,inputs)
        y = T.clip(y, 0.001, 0.999) # stability 
        
        # loss terms
        logdets = sum([get_output(logdet,ep) for logdet in logdets_layers])
        logqw = - (0.5*(ep**2).sum(1) + 0.5*T.log(2*np.pi)*num_params + logdets)
        #logpw = log_normal(weights,0.,-T.log(lbda)).sum(1)
        logpw = log_stdnormal(weights).sum(1)
        kl = (logqw - logpw).mean()
        logpyx = - cc(y,target_var).mean()
        loss = - (logpyx - kl/T.cast(dataset_size,floatX))
        params = lasagne.layers.get_all_params([h_layer,layer])

    else:
        # filler
        h_layer = lasagne.layers.InputLayer([None, 784])
        # JUST primary net
        layer = lasagne.layers.InputLayer([None,784])
        inputs = {layer:input_var}
        for nn, ws in enumerate(weight_shapes):
            layer = lasagne.layers.DenseLayer(layer, ws[1], nonlinearity=nonlinearity)
            if nn < len(weight_shapes)-1 and model in ['dropout', 'dropout2']:
                layer = lasagne.layers.dropout(layer, .5)
            print layer.output_shape
        layer.nonlinearity = nonlinearities.softmax
        y = get_output(layer,inputs)
        y = T.clip(y, 0.001, 0.999) # stability 
        loss = cc(y,target_var).mean()
        params = lasagne.layers.get_all_params([h_layer,layer])
        loss = loss + lasagne.regularization.l2(flatten_list(params)) * np.float32(1.e-5)

    
    # TRAIN FUNCTION
    grads = T.grad(loss, params)
    mgrads = lasagne.updates.total_norm_constraint(grads,
                                                   max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
    updates = lasagne.updates.adam(cgrads, params, 
                                   learning_rate=lr)
                                        
    train = theano.function([input_var,target_var,dataset_size,lr],
                            loss,updates=updates,
                            on_unused_input='warn')
    predict = theano.function([input_var],y.argmax(1))
    predict_probs = theano.function([input_var],y)

    # TODO: don't redefine :P 
    def MCpred(X, inds=None, num_samples=10, returns='preds'):
        from utils import MCpred
        return MCpred(X, predict_probs_fn=predict_probs, num_samples=num_samples, inds=inds, returns=returns)



    ##################
    # TRAIN
    X, Y = train_x[:size],train_y[:size]
    Xt, Yt = valid_x,valid_y
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
    records={}
    records['loss'] = []
    records['acc'] = []
    records['val_acc'] = []
    
    t = 0
    import time
    t0 = time.time()
    for e in range(epochs):
        
        if lrdecay:
            lr = lr0 * 10**(-e/float(epochs-1))
        else:
            lr = lr0         
            
        for i in range(N/bs):
            x = X[i*bs:(i+1)*bs]
            y = Y[i*bs:(i+1)*bs]
            
            loss = train(x,y,N,lr)
            
            #if t%100==0:
            if i == 0:# or t>8000:
                print 'time', time.time() - t0
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)
                tr_inds = np.random.choice(len(X), num_examples, replace=False)
                te_inds = np.random.choice(len(Xt), num_examples, replace=False)
                tr_acc = (MCpred(X, inds=tr_inds)==Y[tr_inds].argmax(1)).mean()
                te_acc = (MCpred(Xt, inds=te_inds)==Yt[te_inds].argmax(1)).mean()
                #assert False
                print '\ttrain acc: {}'.format(tr_acc)
                print '\ttest acc: {}'.format(te_acc)
                records['loss'].append(loss)
                records['acc'].append(tr_acc)
                records['val_acc'].append(te_acc)
                if save:
                    np.save(save_path + '_records.npy', records)
                    np.save(save_path + '_params.npy', lasagne.layers.get_all_param_values([h_layer, layer]))
                    if records['val_acc'][-1] == np.max(records['val_acc']):
                        np.save(save_path + '_params_best.npy', lasagne.layers.get_all_param_values([h_layer, layer]))

            t+=1

if save:
    # load best and do proper evaluation
    lasagne.layers.set_all_param_values([h_layer, layer], np.load(save_path + '_params_best.npy'))
    best_acc = (MCpred(Xt, inds=range(len(Xt)), num_samples=100) == Yt.argmax(1)).mean()
    np.save(save_path + '_best_val_acc=' + str(np.round(100*best_acc, 2)) + '.npy', best_acc)

    if test_eval: # TEST SET
        Xt, Yt = test_x, test_y
        best_acc = (MCpred(Xt, inds=range(len(Xt)), num_samples=100) == Yt.argmax(1)).mean()
        np.save(save_path + '_best_test_acc=' + str(np.round(100*best_acc, 2)) + '.npy', best_acc)

        





# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# --------------------------------------------
# Anomaly Detection Metrics
if 1:
    print "                                                                           RUNNING ANOMALY DETECTION EVALUATIONS!"

        
    import time
    t0 = time.time()
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams

    noise_level=1.


    def noised(dset, lvl, type='Gaussian'):
        #return dset + (lvl * np.random.randn(*dset.shape)).astype('float32') # <-- add instead
        if type == 'uniform':
            return (lvl * np.random.rand(*dset.shape)).astype('float32')
        else:
            return (lvl * np.random.randn(*dset.shape)).astype('float32')

    ##################################################
    # from https://github.com/hendrycks/error-detection/blob/master/Vision/MNIST_Abnormality_Module.ipynb
    print "load notMNIST, CIFAR-10, and Omniglot"
    try:
        notmnist_dataset = np.load('not_mnist.npy')
        cifar_batch = np.load('CIFAR10-bw.npy')
        omni_images = np.load('omniglot.npy')
        print "loaded saved npy files"
    except:
        import pickle
        pickle_file = './data/notMNIST.pickle'
        with open(pickle_file, 'rb') as f:
            #save = pickle.load(f, encoding='latin1')
            save = pickle.load(f)
            notmnist_dataset = save['test_dataset'].reshape((-1, 28 * 28))
            del save

        np.save('not_mnist.npy', notmnist_dataset)

        from helpers import load_data10
        _, _, X_test, _ = load_data10()
        import tensorflow as tf
        try:
            sess = tf.Session()
            with sess.as_default():
                cifar_batch = sess.run(tf.image.resize_images(tf.image.rgb_to_grayscale(X_test), (28, 28))).reshape((-1, 784))
        except:
            pass

        np.save('CIFAR10-bw.npy', cifar_batch)

        import scipy.io as sio
        import scipy.misc as scimisc
        # other alphabets have characters which overlap
        safe_list = [0,2,5,6,8,12,13,14,15,16,17,18,19,21,26]
        m = sio.loadmat("./data/data_background.mat")

        squished_set = []
        for safe_number in safe_list:
            for alphabet in m['images'][safe_number]:
                for letters in alphabet:
                    for letter in letters:
                        for example in letter:
                            squished_set.append(scimisc.imresize(1 - example[0], (28,28)).reshape(1, 28*28))

        omni_images = np.concatenate(squished_set, axis=0)
        np.save('omniglot.npy', omni_images)

    print "done loading notMNIST, CIFAR-10, and Omniglot"
    ################################################################
    ################################################################
    #######################



    # TODO: score functions

    # TODO: get_results CONFIDENCE

    #####################
    from sklearn.metrics import roc_auc_score as roc
    from sklearn.metrics import average_precision_score as pr
    def get_results(ins, oos): #in/out of sample
        """
        returns AOROC, AOPR (success), AOPR (failure) 
        """
        rval = []
        y_true = np.hstack((np.ones(len(ins)), np.zeros(len(oos))))
        y_score = np.hstack((ins, oos))
        rval += [round(roc(y_true, y_score)*100, 2),
                round(pr(y_true, y_score)*100, 2)]
        y_true = np.hstack((np.zeros(len(ins)), np.ones(len(oos))))
        y_score = -y_score
        rval += [#round(roc(y_true, y_score)*100, 2),
                round(pr(y_true, y_score)*100, 2)]
        return rval


    #######################
    # IDEAS FOR y_score:
    """
    baseline
    entropy of estimated Dirichlet?
    avg entropy of predictive distribution
    avg entropy of predictions (along each axis)
    entropy of flattened array
    standard deviation (something something...)
    """

    score_fns = []

    # These score functions are NEGATIVEs of the acquisition functions they are named for
    # SAMPLES: nsamples, nexamples, nclasses
    # returns scores: nexamples
    # they are listed in alphabetic order
    import scipy.stats
    def bald(samples):
        return scipy.stats.entropy(samples.mean(0).T) - scipy.stats.entropy(samples.transpose(2,0,1)).mean(0)
    score_fns.append(bald)
    def max_ent(samples):
        return scipy.stats.entropy(samples.mean(0).T)
    score_fns.append(max_ent)
    if 0: 
        def min_margin(samples): # TODO
            return scipy.stats.entropy(samples.mean(0).T)
        score_fns.append(min_margin)
    def mean_std(samples):
        stds = np.maximum(((samples**2).mean(0) - samples.mean(0)**2),0)**.5
        return stds.mean(-1)
    score_fns.append(mean_std)
    def var_ratio(samples):
        return samples.mean(0).max(-1)
    score_fns.append(var_ratio)




    #######################
    # get rid of clipping for maximum performance!
    y = get_output(layer,inputs)
    #y = T.clip(y, 0.001, 0.999)
    probs = theano.function([input_var],y)


    #######################
    # predictions on clean data

    from utils import MCpred
    clean_samples = MCpred(Xt, predict_probs_fn=probs, num_samples=100, returns='samples')
    clean_probs = clean_samples.mean(0)
    clean_preds = clean_probs.argmax(-1)

    oods = []
    oods.append( noised(Xt, noise_level, 'uniform'))
    oods.append( omni_images )
    oods.append( cifar_batch )
    oods.append( noised(Xt, noise_level) )
    oods.append( notmnist_dataset )

    # RESULTS ARE IN THE SAME ORDER AS IN THE TABLES IN DAN's paper

    ##########################
    # error-detection
    print "\n Error detection"
    err_results= np.empty((len(score_fns), 3))
    for nscore, score_fn in enumerate(score_fns):
        print "Error detection", nscore
        gtruth = np.argmax(Yt, axis=-1)
        is_correct = np.equal(clean_preds, gtruth)
        correct = clean_samples[:, is_correct]
        incorrect = clean_samples[:, np.logical_not(is_correct)]
        clean_scores = score_fn(correct)
        # check that all score functions return a score for each example (i.e. have the right shape)
        assert len(clean_scores) == correct.shape[1]
        err_scores = score_fn(incorrect)
        err_results[nscore][:3] = get_results(clean_scores, err_scores)
    if save:
        if test_eval:
            np.save(save_path + '_test_err_results.npy', err_results)
        else:
            np.save(save_path + '_val_err_results.npy', err_results)
            

    ##########################
    # OOD-detection
    print "\nOOD detection"
    ood_results= np.empty((len(score_fns), len(oods), 3))
    for nscore, score_fn in enumerate(score_fns):
        for nood, ood in enumerate(oods):
            print "OOD detection", nscore, nood
            clean_scores = score_fn(clean_samples)
            ood_samples = MCpred(ood, num_samples=100, returns='samples')
            ood_scores = score_fn(ood_scores)
            ood_results[nscore][nood] = get_results(clean_scores, ood_scores)
    if save:
        if test_eval:
            np.save(save_path + '_test_ood_results.npy', ood_results)
        else:
            np.save(save_path + '_val_ood_results.npy', ood_results)


    print "                                                                                        DONE,   total time=", time.time() - t0
            










if 0: # DEPRECATED
    ##########################
    # OOD-detection
    import collections
    risky = collections.OrderedDict() # easy-to-hard
    risky['uniform'] = noised(Xt, noise_level, 'uniform')
    risky['omniglot'] = omni_images
    risky['CIFARbw'] = cifar_batch
    risky['normal'] = noised(Xt, noise_level)
    risky['notMNIST'] = notmnist_dataset

    for kk, vv in risky.items():
        print "\n",kk
        nyht = np.zeros((num_samples, num_examples, 10))
        for ind in range(num_samples):
            nyht[ind] = probs(vv[:num_examples])
        print "done sampling!"
        nyht = np.mean(nyht, axis=0)
        print get_AOC(yhtm, nyht)

    # samples: nsample, nexample, 10
    def baseline(samples):
        return np.mean(samples, 0)

    def entropy():
        pass


    if 0:

        # TODO: how diverse are the samples? (how to evaluate that?? what to compare to / expect??)

        # 2D scatter-plots of sampled params
        for i in range(9):                                                                                                                     
            subplot(3,3,i+1)
            seaborn.regplot(thet[:, np.random.choice(7940)], thet[:, np.random.choice(7940)]) 
        # look at actual correlation coefficients
        hist([scipy.stats.pearsonr(thet[:, np.random.choice(7940)], thet[:, np.random.choice(7940)])[1] for _ in range(10000)], 100) 

        # TODO: what does the posterior over parameters look like? (we expect to see certain dependencies... e.g. in the simplest case...)
        #   So we can actually see that easily in a toy example, where output = a*b*input, so we just need a*b to equal the right thing, and we can compute the exact posterior based on #examples, etc... and then we can see the difference between independent and not

