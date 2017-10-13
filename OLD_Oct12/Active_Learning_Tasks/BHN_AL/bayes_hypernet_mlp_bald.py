#!/usr/bin/env python
from BHNs import MLPWeightNorm_BHN
from ops import load_mnist
from utils import log_normal, log_laplace
import numpy as np
import random
random.seed(5001)

def split_train_pool_data(X_train, y_train, X_val, y_val):

    random_split = np.asarray(random.sample(range(0,X_train.shape[0]), X_train.shape[0]))

    X_train = X_train[random_split, :]
    y_train = y_train[random_split]

    X_valid = X_train[10000:15000, :]
    y_valid = y_train[10000:15000]

    X_pool = X_train[20000:60000, :]
    y_pool = y_train[20000:60000]


    X_train = X_train[0:10000, :]
    y_train = y_train[0:10000]


    return X_train, y_train, X_valid, y_valid, X_pool, y_pool

def get_initial_training_data(X_train_All, y_train_All):
    #training data to have equal distribution of classes
    idx_0 = np.array( np.where(y_train_All==0)  ).T
    idx_0 = idx_0[0:2,0]
    X_0 = X_train_All[idx_0, :]
    y_0 = y_train_All[idx_0]

    idx_1 = np.array( np.where(y_train_All==1)  ).T
    idx_1 = idx_1[0:2,0]
    X_1 = X_train_All[idx_1, :]
    y_1 = y_train_All[idx_1]

    idx_2 = np.array( np.where(y_train_All==2)  ).T
    idx_2 = idx_2[0:2,0]
    X_2 = X_train_All[idx_2, :]
    y_2 = y_train_All[idx_2]

    idx_3 = np.array( np.where(y_train_All==3)  ).T
    idx_3 = idx_3[0:2,0]
    X_3 = X_train_All[idx_3, :]
    y_3 = y_train_All[idx_3]

    idx_4 = np.array( np.where(y_train_All==4)  ).T
    idx_4 = idx_4[0:2,0]
    X_4 = X_train_All[idx_4, :]
    y_4 = y_train_All[idx_4]

    idx_5 = np.array( np.where(y_train_All==5)  ).T
    idx_5 = idx_5[0:2,0]
    X_5 = X_train_All[idx_5, :]
    y_5 = y_train_All[idx_5]

    idx_6 = np.array( np.where(y_train_All==6)  ).T
    idx_6 = idx_6[0:2,0]
    X_6 = X_train_All[idx_6, :]
    y_6 = y_train_All[idx_6]

    idx_7 = np.array( np.where(y_train_All==7)  ).T
    idx_7 = idx_7[0:2,0]
    X_7 = X_train_All[idx_7, :]
    y_7 = y_train_All[idx_7]

    idx_8 = np.array( np.where(y_train_All==8)  ).T
    idx_8 = idx_8[0:2,0]
    X_8 = X_train_All[idx_8, :]
    y_8 = y_train_All[idx_8]

    idx_9 = np.array( np.where(y_train_All==9)  ).T
    idx_9 = idx_9[0:2,0]
    X_9 = X_train_All[idx_9, :]
    y_9 = y_train_All[idx_9]

    X_train = np.concatenate((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9), axis=0 )
    y_train = np.concatenate((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9), axis=0 )


    return X_train, y_train

def train_model(train_func,predict_func,X,Y,Xt,Yt,
                lr0=0.1,lrdecay=1,bs=20,epochs=50):
    
    Y = Y.astype('float32')
    print 'trainset X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape)
    N = X.shape[0]    
    records=list()
    
    t = 0
    for e in range(epochs):
        
        if lrdecay:
            lr = lr0 * 10**(-e/float(epochs-1))
        else:
            lr = lr0         
            
        for i in range(N/bs):
            x = X[i*bs:(i+1)*bs]
            y = Y[i*bs:(i+1)*bs]
            
            loss = train_func(x,y,N,lr)
            
            if t%100==0:
                print 'epoch: {} {}, loss:{}'.format(e,t,loss)
                tr_acc = (predict_func(X)==Y.argmax(1)).mean()
                te_acc = (predict_func(Xt)==Yt.argmax(1)).mean()
                print '\ttrain acc: {}'.format(tr_acc)
                # print '\ttest acc: {}'.format(te_acc)
            t+=1
            
        records.append(loss)
        
    return records


def test_model(predict_proba, X_test, y_test):
    mc_samples = 1000
    y_pred_all = np.zeros((mc_samples, X_test.shape[0], 10))

    for m in range(mc_samples):
        y_pred_all[m] = predict_proba(X_test)

    y_pred = y_pred_all.mean(0).argmax(-1)
    y_test = y_test.argmax(-1)

    test_accuracy = np.equal(y_pred, y_test).mean()
    return test_accuracy




def active_learning(acquisition_iterations):

    bh_iterations = 100
    nb_classes = 10
    Queries = 10
    all_accuracy = 0


    filename = '../../mnist.pkl.gz'
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)


    train_x, train_y, valid_x, valid_y, pool_x, pool_y = split_train_pool_data(train_x, train_y, valid_x, valid_y)
    train_x, train_y = get_initial_training_data(train_x, train_y)

    print ("Training Set Size", train_x.shape)

    model = MLPWeightNorm_BHN(lbda=lbda,
                              perdatapoint=perdatapoint,
                              prior=prior,
                              coupling=coupling)
    
    
    recs = train_model(model.train_func,model.predict,
                       train_x[:size],train_y[:size],
                       valid_x,valid_y,
                       lr0,lrdecay,bs,epochs)
   

    test_accuracy = test_model(model.predict_proba, test_x, test_y)

    print ("Test Accuracy", test_accuracy)

    all_accuracy = test_accuracy



    for i in range(acquisition_iterations):

        print('POOLING ITERATION', i)
        pool_subset = 2000

        pool_subset_dropout = np.asarray(random.sample(range(0,pool_x.shape[0]), pool_subset))

        X_pool_Dropout = pool_x[pool_subset_dropout, :]
        y_pool_Dropout = pool_y[pool_subset_dropout]

        score_All = np.zeros(shape=(X_pool_Dropout.shape[0], nb_classes))
        All_Entropy_BH = np.zeros(shape=X_pool_Dropout.shape[0])


        all_bh_classes = np.zeros(shape=(X_pool_Dropout.shape[0], bh_iterations))


        print ("Bayes Hypernet Posterior Samples")
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


        Pooled_X = X_pool_Dropout[x_pool_index, :]
        Pooled_Y = y_pool_Dropout[x_pool_index] 


        delete_Pool_X = np.delete(pool_x, (pool_subset_dropout), axis=0)
        delete_Pool_Y = np.delete(pool_y, (pool_subset_dropout), axis=0)        

        delete_Pool_X_Dropout = np.delete(X_pool_Dropout, (x_pool_index), axis=0)
        delete_Pool_Y_Dropout = np.delete(y_pool_Dropout, (x_pool_index), axis=0)


        pool_x = np.concatenate((pool_x, X_pool_Dropout), axis=0)
        pool_y = np.concatenate((pool_y, y_pool_Dropout), axis=0)

        train_x = np.concatenate((train_x, Pooled_X), axis=0)
        train_y = np.concatenate((train_y, Pooled_Y), axis=0)

        print ("Training Set Size", train_x.shape)

        recs = train_model(model.train_func,model.predict,
                           train_x[:size],train_y[:size],
                           valid_x,valid_y,
                           lr0,lrdecay,bs,epochs)
   

        test_accuracy = test_model(model.predict_proba, test_x, test_y)   

        print ("Test Accuracy", test_accuracy)

        all_accuracy = np.append(all_accuracy, test_accuracy)


    return all_accuracy


def main():

    num_experiments = 3
    acquisition_iterations = 98
    all_accuracy = np.zeros(shape=(acquisition_iterations+1, num_experiments))
    
    for i in range(num_experiments):
        
        accuracy = active_learning(acquisition_iterations)
        all_accuracy[:, i] = accuracy
        np.save('BH_MLP_bald_all_accuracy.npy', all_accuracy)

    mean_accuracy = np.mean(all_accuracy)

    np.save('BH_MLP_bald_all_accuracy.npy', all_accuracy)
    np.save('BH_MLP_bald_mean_accuracy.npy',mean_accuracy)    



if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # boolean: 1 -> True ; 0 -> False
    parser.add_argument('--coupling',default=0,type=int)  
    parser.add_argument('--perdatapoint',default=0,type=int)
    parser.add_argument('--lrdecay',default=0,type=int)  
    
    parser.add_argument('--lr0',default=0.1,type=float)  
    parser.add_argument('--lbda',default=1,type=float)  
    parser.add_argument('--size',default=10000,type=int)      
    parser.add_argument('--bs',default=20,type=int)  
    parser.add_argument('--epochs',default=50,type=int)
    parser.add_argument('--prior',default='log_normal',type=str)
    args = parser.parse_args()
    print args
    
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
