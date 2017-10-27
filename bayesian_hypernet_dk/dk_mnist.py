#!/usr/bin/env python
import os
import time

import numpy
np = numpy
from scipy.stats import mode

import theano
floatX = theano.config.floatX
import lasagne

#from dk_hyperCNN import MCdropoutCNN
from bayesian_hypernet_dk.riashat_cnn import RiashatCNN as MCdropoutCNN
from bayesian_hypernet_dk.BHNs import HyperCNN

from bayesian_hypernet_dk.helpers import *
from bayesian_hypernet_dk.layers import *

from capy_utils import save_list


"""
I am stripping down this file into a simple script that I can use to test things...

The goal is to figure out why BHNs don't do well on very small datasets, as we see at the start of the AL experiments

"""

# TODO: balanced data
# TODO: tweak init
# TODO: different optimizers?

# TODO: copy to lab machines (interactive!)


def train_epoch(train_func,predict_func,X,Y,Xv,Yv,
                lr0=0.1,lrdecay=1,bs=20):
    """ perform a single epoch of training"""
    N = X.shape[0]    
    for i in range( N/bs + int(N%bs > 0) ):
        x = X[i*bs:(i+1)*bs]
        y = Y[i*bs:(i+1)*bs]
        print x.shape, y.shape
        loss = train_func(x,y,N,lr0)

def test_model(predict_proba, X_test, y_test):
    """ Get the accuracy on a given 'test set' """
    #mc_samples = mc_samples # 20
    y_pred_all = np.zeros((mc_samples, X_test.shape[0], 10))

    for m in range(mc_samples):
        y_pred_all[m] = predict_proba(X_test)

    y_pred = y_pred_all.mean(0).argmax(-1)
    y_test = y_test.argmax(-1)

    test_accuracy = np.equal(y_pred, y_test).mean()
    return test_accuracy





if 1:#__name__ == '__main__':
    
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n_units_h',default=200,type=int)  
    parser.add_argument('--init_noise_level',default=-7,type=float)  
    parser.add_argument('--init_scale_h',default=.0001,type=float)  
    parser.add_argument('--init_scale_p',default=.01,type=float)  
    # boolean: 1 -> True ; 0 -> False
    parser.add_argument('--acq',default='bald',type=str, choices=['numerically_stable_bald', 'bald', 'max_ent', 'var_ratio', 'mean_std', 'random', 'zeros'])
    parser.add_argument('--bs',default=128,type=int)  
    parser.add_argument('--coupling',default=4,type=int)  
    parser.add_argument('--n_epochs',default=500,type=int)
    parser.add_argument('--flow',default='RealNVP',type=str)
    parser.add_argument('--lrdecay',default=0,type=int)  
    parser.add_argument('--lr0',default=0.001,type=float)  
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--mc_samples',default=20,type=int)  
    parser.add_argument('--model_type',default='BHN',type=str)  
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--reinit',default=0,type=int)
    parser.add_argument('--size',default=100,type=int)
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
        print("save_path", save_path + '\n')
        #assert False

    locals().update(args_dict)

    clip_grad = 100
    max_norm = 100

    # SET RANDOM SEED (TODO: rng vs. random.seed)
    if seed is None:
        rng = np.random.randint(2**32 - 1)
    np.random.seed(seed)  # for reproducibility
    rng = numpy.random.RandomState(seed)
    random.seed(seed)

    # --------------------------------------------
    print "\n\n\n-----------------------------------------------------------------------\n\n\n"
    print args

    locals().update(args.__dict__) 

    if not os.path.exists(save_dir):
         os.makedirs(save_dir)
    
    lbda = np.cast['float32'](args.lbda)
    if args.prior=='log_normal':
        prior = log_normal
    elif args.prior=='log_laplace':
        prior = log_laplace


    # get data    TODO: shuffle!
    filename = 'mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
    train_x = train_x.reshape(50000,1,28,28)
    perm = np.random.permutation(range(len(train_x)))
    train_x = train_x[perm]
    train_y = train_y[perm]
    train_x = train_x[:size]
    train_y = train_y[:size]
    valid_x = valid_x.reshape(10000,1,28,28)
    valid_x = valid_x[:size]
    valid_y = valid_y[:size]
    train_y = train_y.astype('float32')
    print "Initial Training Data", train_x.shape

    
    # weight-norm init 
    if reinit:
        init_batch_size = min(64, size)
        init_batch = train_x[:size][-init_batch_size:].reshape(init_batch_size,784)
    else:
        init_batch = None

    # select model
    if model_type == 'BHN':
        model = HyperCNN(lbda=lbda,
                    arch='Riashat',
                         perdatapoint=perdatapoint,
                         prior=prior,
                         coupling=coupling,
                         kernel_width=4,
                         pad='valid',
                         stride=1,
                        flow=flow,
                        init_batch=init_batch)
    elif model_type == 'MLE':
        model = MCdropoutCNN(dropout=0,
                    kernel_width=4,
                        arch='Riashat',
                         pad='valid',
                         stride=1)
    else:
        raise Exception('no model named `{}`'.format(model))
        
    predict_fn = model.predict_proba

    tr_lcs = []
    va_lcs = []

    for epoch in range(n_epochs):
        train_epoch(model.train_func,predict_fn,
                           train_x,train_y,
                           valid_x,valid_y,
                           lr0,lrdecay,bs)
        tr_lcs.append(test_model(predict_fn, train_x, train_y))
        va_lcs.append(test_model(predict_fn, valid_x, valid_y))
        print va_lcs[-1]
   
    
    save_list(save_path + '_tr_lcs', tr_lcs)
    save_list(save_path + '_va_lcs', va_lcs)
