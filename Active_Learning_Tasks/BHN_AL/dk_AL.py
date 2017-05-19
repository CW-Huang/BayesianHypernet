#!/usr/bin/env python
from BHNs import HyperCNN
from ops import load_mnist
from utils import log_normal, log_laplace
import numpy
np = numpy
import random
#random.seed(5001)
from lasagne import layers
from scipy.stats import mode

from dk_hyperCNN import MCdropoutCNN

import time
import os

from AL_helpers import *


# TODO: fixme... learning rates!!

def train_model(train_func,predict_func,X,Y,Xt,Yt,
                lr0=0.1,lrdecay=1,bs=20,epochs=50):

    N = X.shape[0]    
    records=list()
    
    t = 0
    for e in range(epochs):
        
        if lrdecay:
            lr = lr0 * 10**(-e/float(epochs-1))
        else:
            lr = lr0         
            
        #for i in range(N/bs):
        for i in range( N/bs + int(N%bs > 0) ):
            x = X[i*bs:(i+1)*bs]
            y = Y[i*bs:(i+1)*bs]
            
            loss = train_func(x,y,N,lr)
            
            if e == epochs-1:#i==0:#t%100==0:
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)
                tr_acc = (predict_func(X)==Y.argmax(1)).mean()
                te_acc = (predict_func(Xt)==Yt.argmax(1)).mean()
                print '\ttrain acc: {}'.format(tr_acc)
                # print '\ttest acc: {}'.format(te_acc)
            t+=1
            
        records.append(loss)
        
    return records


def test_model(predict_proba, X_test, y_test):
    mc_samples = 100
    y_pred_all = np.zeros((mc_samples, X_test.shape[0], 10))

    for m in range(mc_samples):
        y_pred_all[m] = predict_proba(X_test)

    y_pred = y_pred_all.mean(0).argmax(-1)
    y_test = y_test.argmax(-1)

    test_accuracy = np.equal(y_pred, y_test).mean()
    return test_accuracy




def active_learning(acquisition_iterations):

    t0 = time.time()

    bh_iterations = 100
    nb_classes = 10
    Queries = 10
    all_accuracy = 0

    acquisition_iterations = 98

    filename = '../../mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
    train_x = train_x.reshape(50000,1,28,28)
    valid_x = valid_x.reshape(10000,1,28,28)
    test_x = test_x.reshape(10000,1,28,28)
        
    train_x, train_y, pool_x, pool_y = split_train_pool_data(train_x, train_y)

    train_y_multiclass = train_y.argmax(1)


    train_x, train_y = get_initial_training_data(train_x, train_y_multiclass)

    print "Initial Training Data", train_x.shape

    # select model
    if arch == 'hyperCNN':
        model = HyperCNN(lbda=lbda,
                         perdatapoint=perdatapoint,
                         prior=prior,
                         coupling=coupling,
                         kernel_width=4,
                         pad='valid',
                         stride=1,
                         extra_linear=extra_linear)
                         #dataset=dataset)
    elif arch == 'CNN':
        model = MCdropoutCNN(kernel_width=4,
                         pad='valid',
                         stride=1)
    elif arch == 'CNN_spatial_dropout':
        model = MCdropoutCNN(dropout='spatial',
                         kernel_width=4,
                         pad='valid',
                         stride=1)
    elif arch == 'CNN_dropout':
        model = MCdropoutCNN(dropout=1,
                         kernel_width=4,
                         pad='valid',
                         stride=1)
    else:
        raise Exception('no model named `{}`'.format(model))
        

    
    train_y = train_y.astype('float32')
    recs = train_model(model.train_func,model.predict,
                       train_x[:size],train_y[:size],
                       valid_x,valid_y,
                       lr0,lrdecay,bs,epochs)
   
    test_accuracy = test_model(model.predict_proba, test_x, test_y)

    print "                                                          Test Accuracy", test_accuracy

    all_accuracy = test_accuracy

    for i in range(acquisition_iterations):

        print'time', time.time() - t0
    	print'POOLING ITERATION', i
    	pool_subset = pool_size

    	pool_subset_dropout = np.asarray(random.sample(range(0,pool_x.shape[0]), pool_subset))

    	X_pool_Dropout = pool_x[pool_subset_dropout, :, :, :]
    	y_pool_Dropout = pool_y[pool_subset_dropout]



        #####################################3
        # BEGIN ACQUISITION
        if acq == 'bald':
    	    score_All = np.zeros(shape=(X_pool_Dropout.shape[0], nb_classes))
            All_Entropy_BH = np.zeros(shape=X_pool_Dropout.shape[0])
            all_bh_classes = np.zeros(shape=(X_pool_Dropout.shape[0], bh_iterations))


            for d in range(bh_iterations):
                bh_score = model.predict_proba(X_pool_Dropout)
                score_All = score_All + bh_score

                bh_score_log = np.log2(bh_score)
                Entropy_Compute = - np.multiply(bh_score, bh_score_log)

                Entropy_Per_BH = np.sum(Entropy_Compute, axis=1)

                All_Entropy_BH = All_Entropy_BH + Entropy_Per_BH

                bh_classes = np.max(bh_score, axis=1)
                all_bh_classes[:, d] = bh_classes



            ### for plotting uncertainty
            predicted_class = np.max(all_bh_classes, axis=1)
            predicted_class_std = np.std(all_bh_classes, axis=1)

            Avg_Pi = np.divide(score_All, bh_iterations)
            Log_Avg_Pi = np.log2(Avg_Pi)
            Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
            Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

            G_X = Entropy_Average_Pi

            Average_Entropy = np.divide(All_Entropy_BH, bh_iterations)
            F_X = Average_Entropy
            U_X = G_X - F_X
            sort_values = U_X.flatten()
            x_pool_index = sort_values.argsort()[-Queries:][::-1]
            #print x_pool_index.shape # 10
            #assert False

        elif acq == 'max_ent':
    	    score_All = np.zeros(shape=(X_pool_Dropout.shape[0], nb_classes))
            for d in range(bh_iterations):
                bh_score = model.predict_proba(X_pool_Dropout)
                score_All = score_All + bh_score

            Avg_Pi = np.divide(score_All, bh_iterations)
            Log_Avg_Pi = np.log2(Avg_Pi)
            Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
            Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

            U_X = Entropy_Average_Pi
            sort_values = U_X.flatten()
            x_pool_index = sort_values.argsort()[-Queries:][::-1]

        elif acq == 'var_ratio':
            All_BH_Classes = np.zeros(shape=(X_pool_Dropout.shape[0],1))

            for d in range(bh_iterations):
                bh_score = model.predict(X_pool_Dropout)
                bh_score = np.array([bh_score]).T
                All_BH_Classes = np.append(All_BH_Classes, bh_score, axis=1)


            Variation = np.zeros(shape=(X_pool_Dropout.shape[0]))

            for t in range(X_pool_Dropout.shape[0]):
                L = np.array([0])
                for d_iter in range(bh_iterations):
                    L = np.append(L, All_BH_Classes[t, d_iter+1])                      
                Predicted_Class, Mode = mode(L[1:])
                v = np.array(  [1 - Mode/float(bh_iterations)])
                Variation[t] = v     

            sort_values = Variation.flatten()
            x_pool_index = sort_values.argsort()[-Queries:][::-1]

        elif acq == 'mean_std':
            All_Dropout_Scores = np.zeros(shape=(X_Pool_Dropout.shape[0], nb_classes))

            for d in range(dropout_iterations):
                dropout_score = model.predict_stochastic(X_Pool_Dropout,batch_size=batch_size, verbose=1)
                All_Dropout_Scores = np.append(All_Dropout_Scores, dropout_score, axis=1)

            All_Std = np.zeros(shape=(X_Pool_Dropout.shape[0],nb_classes))
            BayesSegnet_Sigma = np.zeros(shape=(X_Pool_Dropout.shape[0],1)) 

            for t in range(X_Pool_Dropout.shape[0]):
                for r in range(nb_classes):
                    L = np.array([0])
                    L = np.append(L, All_Dropout_Scores[t, r+10])
                    
                    L_std = np.std(L[1:])
                    All_Std[t,r] = L_std
                    E = All_Std[t,:]
                    BayesSegnet_Sigma[t,0] = sum(E)

            a_1d = BayesSegnet_Sigma.flatten()
            x_pool_index = a_1d.argsort()[-Queries:][::-1]
            
        elif acq == 'random':
            x_pool_index = np.asarray(random.sample(range(0, 38000), Queries))
            #x_pool_index = np.random.choice(range(pool_size), Queries, replace=False)


        # END ACQUISITION
        #####################################3


        Pooled_X = X_pool_Dropout[x_pool_index, :, :, :]
        Pooled_Y = y_pool_Dropout[x_pool_index] 
        delete_Pool_X = np.delete(pool_x, (pool_subset_dropout), axis=0)
        delete_Pool_Y = np.delete(pool_y, (pool_subset_dropout), axis=0)        
        delete_Pool_X_Dropout = np.delete(X_pool_Dropout, (x_pool_index), axis=0)
        delete_Pool_Y_Dropout = np.delete(y_pool_Dropout, (x_pool_index), axis=0)
        pool_x = np.concatenate((pool_x, X_pool_Dropout), axis=0)
        pool_y = np.concatenate((pool_y, y_pool_Dropout), axis=0)
        train_x = np.concatenate((train_x, Pooled_X), axis=0)
        train_y = np.concatenate((train_y, Pooled_Y), axis=0).astype('float32')
        #print pool_x.shape, Pooled_X.shape, train_x.shape
        #assert False


        if 0:# don't warm start (TODO!)
            model = HyperCNN(lbda=lbda,
                              perdatapoint=perdatapoint,
                              prior=prior,
                              kernel_width=4,
                              pad='valid',
                              stride=1,
                              coupling=coupling,
                              extra_linear=extra_linear)
    
        recs = train_model(model.train_func,model.predict,
	                       train_x[:size],train_y[:size],
	                       valid_x,valid_y,
	                       lr0,lrdecay,bs,epochs)
   

        test_accuracy = test_model(model.predict_proba, test_x, test_y)   

        print "                                                          Test Accuracy", test_accuracy

        all_accuracy = np.append(all_accuracy, test_accuracy)


    return all_accuracy


def main():

    #num_experiments = 3
    acquisition_iterations = 98
    all_accuracy = np.zeros(shape=(acquisition_iterations+1, num_experiments))
    
    for i in range(num_experiments):
        
        accuracy = active_learning(acquisition_iterations)
        all_accuracy[:, i] = accuracy
        np.save(save_path + '_all_accuracy.npy', all_accuracy)

    
    mean_accuracy = np.mean(all_accuracy)

    np.save(save_path + '_all_accuracy.npy'), all_accuracy)
    np.save(save_path + '_mean_accuracy.npy'), mean_accuracy)



if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # boolean: 1 -> True ; 0 -> False
    parser.add_argument('--acq',default='bald',type=str, choices=['bald', 'max_ent', 'var_ratio', 'mean_std', 'random'])
    parser.add_argument('--arch',default='hyperCNN',type=str)
    parser.add_argument('--bs',default=128,type=int)  
    parser.add_argument('--coupling',default=4,type=int)  
    parser.add_argument('--epochs',default=50,type=int)
    parser.add_argument('--lrdecay',default=0,type=int)  
    parser.add_argument('--lr0',default=0.0001,type=float)  
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--num_experiments',default=3,type=int)  
    parser.add_argument('--params_reset',default='none', type=str, choices=['deterministic', 'random', 'none'] ) # TODO
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    parser.add_argument('--pool_size',default=2000,type=int)
    parser.add_argument('--size',default=10000,type=int)      
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
    print "\n\n\n-----------------------------------------------------------------------\n\n\n"
    print args
    

    locals().update(args.__dict__) 

    if not os.path.exists(save_dir):
         os.makedirs(save_dir)
    

    acq = args.acq
    coupling = args.coupling
    perdatapoint = args.perdatapoint
    lrdecay = args.lrdecay
    lr0 = args.lr0
    lbda = np.cast['float32'](args.lbda)
    bs = args.bs
    epochs = args.epochs
    if args.prior=='log_normal':
        prior = log_normal
    elif args.prior=='log_laplace':
        prior = log_laplace
    size = max(10,min(50000,args.size))
    

    main()
