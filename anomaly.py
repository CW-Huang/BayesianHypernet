"""
WIP

run this in ipython after running:
    dk_mnist_mlp_weightnorm.py
with the command:
    run -i anomaly.py

This script runs the MNIST anomaly detection baselines from:
https://arxiv.org/pdf/1610.02136.pdf

TODO: everything except Gaussian noise

"""


import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

np.random.seed(427)

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--noise_level',default=.5,type=float)  
parser.add_argument('--num_examples',default=1000,type=int)  
parser.add_argument('--num_samples',default=100,type=int)  
args = parser.parse_args()
locals().update(args.__dict__)
print args


# from https://github.com/hendrycks/error-detection/blob/master/Vision/MNIST_Abnormality_Module.ipynb
# load notMNIST, CIFAR-10, and Omniglot
import pickle
pickle_file = './data/notMNIST.pickle'
with open(pickle_file, 'rb') as f:
    #save = pickle.load(f, encoding='latin1')
    save = pickle.load(f)
    notmnist_dataset = save['test_dataset'].reshape((-1, 28 * 28))
    del save

from load_cifar10 import load_data10
_, _, X_test, _ = load_data10()
# TODO:
#cifar_batch = sess.run(tf.image.resize_images(tf.image.rgb_to_grayscale(X_test), 28, 28))

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
print "done loading notMNIST, CIFAR-10, and Omniglot"
################33


# TODO: add noise vs. replace w/noise
def noised(dset, lvl):
    #return dset + (lvl * np.random.randn(*dset.shape)).astype('float32')
    return (lvl * np.random.randn(*dset.shape)).astype('float32')

# eval function
y = T.clip(get_output(layer,inputs), 0.001, 0.999)
probs = theano.function([input_var],y)

# predictions on clean data
X, Y = train_x, train_y
Xt, Yt = valid_x, valid_y

yh = np.zeros((num_samples, num_examples, 10))
yht = np.zeros((num_samples, num_examples, 10))
for ind in range(num_samples):
    yh[ind] = probs(X[:num_examples])
    yht[ind] = probs(Xt[:num_examples])
# TODO: don't overwrite!
yh = np.mean(yh, axis=0)
yht = np.mean(yht, axis=0)


# TODO
#for risky in ['notMNIST', 'CIFARbw', 'omniglot', 'normal', 'uniform']:
    

# TODO: train models like Hendrycks' architecture
for noise_level in [1.]:#[.01, .1, .2, .5, 1., 2., 5., 10., 20., 50.]:
    print "\nnoise_level=", str(noise_level)
    nX = noised(X, noise_level)
    nXt = noised(Xt, noise_level)
    

    # predictions on noised data
    nyh = np.zeros((num_samples, num_examples, 10))
    nyht = np.zeros((num_samples, num_examples, 10))
    for ind in range(num_samples):
        nyh[ind] = probs(nX[:num_examples])
        nyht[ind] = probs(nXt[:num_examples])

    print "done sampling!"
    # TODO: don't overwrite!
    nyh = np.mean(nyh, axis=0)
    nyht = np.mean(nyht, axis=0)

    from sklearn.metrics import roc_auc_score as roc
    from sklearn.metrics import average_precision_score as pr
    def get_AOC(ins, oos): #in/out of sample
        rval = []
        y_true = np.hstack((np.ones(len(ins)), np.zeros(len(oos))))
        y_score = np.vstack((ins, oos))
        y_score = y_score.max(axis=1)
        rval += [round(roc(y_true, y_score)*100, 2),
               round(pr(y_true, y_score)*100, 2)]
        y_true = np.hstack((np.zeros(len(ins)), np.ones(len(oos))))
        y_score = -y_score
        rval += [#round(roc(y_true, y_score)*100, 2),
               round(pr(y_true, y_score)*100, 2)]
        return rval

    print "train source/target, valid source/target:"
    print get_AOC(yh, yh)
    print get_AOC(yh, nyh)
    print get_AOC(yht, yht)
    print get_AOC(yht, nyht)




##############33333

if 0:
    print "train perf=", np.equal(np.argmax(MCs.mean(0), -1), np.argmax(train_y[:1000], -1)).mean()
    print "valid perf=", np.equal(np.argmax(vMCs.mean(0), -1), np.argmax(valid_y[:1000], -1)).mean()

    # TODO: how diverse are the samples? (how to evaluate that?? what to compare to / expect??)

    # 2D scatter-plots of sampled params
    for i in range(9):                                                                                                                     
        subplot(3,3,i+1)
        seaborn.regplot(thet[:, np.random.choice(7940)], thet[:, np.random.choice(7940)]) 
    # look at actual correlation coefficients
    hist([scipy.stats.pearsonr(thet[:, np.random.choice(7940)], thet[:, np.random.choice(7940)])[1] for _ in range(10000)], 100) 


    # TODO: what does the posterior over parameters look like? (we expect to see certain dependencies... e.g. in the simplest case...)
    #   So we can actually see that easily in a toy example, where output = a*b*input, so we just need a*b to equal the right thing, and we can compute the exact posterior based on #examples, etc... and then we can see the difference between independent and not

