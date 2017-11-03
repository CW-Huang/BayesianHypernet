#!/usr/bin/env python

"""
WIP

This script implements the idea of sequential Bayesian updating for BDNNs with VI approximate posteriors.
"""

from ops import load_mnist
from utils import log_normal, log_laplace, train_model, evaluate_model

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
floatX = theano.config.floatX

# From BHNs.py
import lasagne
from lasagne import nonlinearities
rectify = nonlinearities.rectify
softmax = nonlinearities.softmax
from lasagne.objectives import categorical_crossentropy as cc
from lasagne.objectives import squared_error as se

from helpers import flatten_list
# ---------------------------------------------------------------
# Define new functions/classes
# TODO: can compare with proper Bayesian updating (??)

def KL(prior_mean, prior_log_var, posterior_mean, posterior_log_var, delta=.001):
    """
    Compute KL between Gaussian posterior and prior
    equation taken from https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    prior_var = T.exp(prior_log_var) + delta
    posterior_var = T.exp(posterior_log_var) + delta
    return .5 * (T.log(prior_var) - T.log(posterior_var) + (posterior_var + (prior_mean - posterior_mean)**2) / prior_var  - 1)

def ts(size, scale=.000001, bias=0):
    return theano.shared((scale * rng.randn(size) + bias).astype(floatX))

class Full_BHN(object):
    """
    hypernet (really just BbB/mean field for now) that outputs ALL the primary net parameters (including biases!!)
    """
    def __init__(self,
                 srng = RandomStreams(seed=427),
                 prior_mean = 0,
                 prior_log_var = 0,
                 n_hiddens=2,
                 n_units=800,
                 n_inputs=784,
                 n_classes=10,
                 output_type = 'categorical',
                 random_biases=1,
                 #dataset_size=None,
                 opt='adam',
                 #weight=1.,# the weight of the KL term
                 **kargs):
        
        self.__dict__.update(locals())

        # TODO
        self.dataset_size = T.scalar('dataset_size')
        self.weight = T.scalar('weight')
        self.learning_rate = T.scalar('learning_rate')

        self.weight_shapes = []
        self.weight_shapes = []
        if n_hiddens > 0:
            self.weight_shapes.append((n_inputs,n_units))
            #self.params.append((theano.shared()))
            for i in range(1,n_hiddens):
                self.weight_shapes.append((n_units,n_units))
            self.weight_shapes.append((n_units,n_classes))
        else:
            self.weight_shapes = [(n_inputs, n_classes)]

        if self.random_biases:
            self.num_params = sum((ws[0]+1)*ws[1] for ws in self.weight_shapes)
        else:
            self.num_params = sum((ws[0])*ws[1] for ws in self.weight_shapes)
        
        self.wd1 = 1
        self.X = T.matrix()
        self.y = T.matrix()
        self.mean = ts(self.num_params)
        self.log_var = ts(self.num_params, scale=1e-6, bias=-1e8)
        self.params = [self.mean, self.log_var]
        self.ep = self.srng.normal(size=(self.num_params,), dtype=floatX)
        self.weights = self.mean + (T.exp(self.log_var) + np.float32(.000001)) * self.ep

        t = 0
        acts = self.X
        for nn, ws in enumerate(self.weight_shapes):
            if self.random_biases:
                num_param = (ws[0]+1) * ws[1]
                weight_and_bias = self.weights[t:t+num_param]
                weight = weight_and_bias[:ws[0]*ws[1]].reshape((ws[0], ws[1]))
                bias = weight_and_bias[ws[0]*ws[1]:].reshape((ws[1],))
                acts = T.dot(acts, weight) + bias
            else:
                assert False # TODO
            if nn < len(self.weight_shapes) - 1:
                acts = (acts> 0.) * (acts)
            else:
                acts = T.nnet.softmax(acts)

            t += num_param
            
        y_hat = acts
        #y_hat = T.clip(y_hat, 0.001, 0.999) # stability
        self.y_hat = y_hat

        self.kl = KL(self.prior_mean, self.prior_log_var, self.mean, self.log_var).sum(-1).mean()
        self.logpyx = - cc(self.y_hat, self.y).mean()
        self.logpyx = - se(self.y_hat, self.y).mean()
        self.loss = - (self.logpyx - self.weight * self.kl/T.cast(self.dataset_size,floatX))
        self.loss = se(self.y_hat, self.y).mean()
        self.logpyx_grad = flatten_list(T.grad(-self.logpyx, self.params, disconnected_inputs='warn')).norm(2)
        self.monitored = [self.logpyx, self.logpyx_grad, self.kl]

        #def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.X],self.y_hat)
        self.predict = theano.function([self.X],self.y_hat.argmax(1))
        self.predict_fixed_mask = theano.function([self.X, self.weights],self.y_hat)
        self.sample_weights = theano.function([], self.weights)
        self.monitor_fn = theano.function([self.X, self.y], self.monitored)#, (self.predict(x) == y).sum()
        
        #def _get_grads(self):
        grads = T.grad(self.loss, self.params)
        #mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=self.max_norm)
        #cgrads = [T.clip(g, -self.clip_grad, self.clip_grad) for g in mgrads]
        cgrads = grads
        if self.opt == 'adam':
            self.updates = lasagne.updates.adam(cgrads, self.params, 
                                                learning_rate=self.learning_rate)
        elif self.opt == 'momentum':
            self.updates = lasagne.updates.nesterov_momentum(cgrads, self.params, 
                                                learning_rate=self.learning_rate)
        elif self.opt == 'sgd':
            self.updates = lasagne.updates.sgd(cgrads, self.params, 
                                                learning_rate=self.learning_rate)
                                    
        #def _get_train_func(self):
        inputs = [self.X,
                  self.y,
                  self.dataset_size,
                  self.learning_rate,
                  self.weight]
        train = theano.function(inputs,
                                self.loss,updates=self.updates,
                                on_unused_input='warn')
        self.train_func_ = train
        # DK - putting this here, because is doesn't get overwritten by subclasses
        self.monitor_func = theano.function([self.X,
                                 self.y,
                                 self.dataset_size,
                                 self.learning_rate],
                                self.monitored,
                                on_unused_input='warn')
    
    def train_func(self,x,y,n,lr=.001,w=1.0):
        return self.train_func_(x,y,n,lr,w)

    def monitor_fn(self, x, y):
        rval = self.monitor_fn(x,y)
        print "logpyx, kl, acc = ", rval
        return rval 
    

# ---------------------------------------------------------------
import argparse
import os
import sys
import numpy 
np = numpy

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--lr', type=float, default=.001)
#
parser.add_argument('--n_epochs', type=int, default=11)
parser.add_argument('--n_hiddens', type=int, default=1)
parser.add_argument('--n_units', type=int, default=50)
parser.add_argument('--n_splits', type=int, default=1)
parser.add_argument('--n_train', type=int, default=2000)
parser.add_argument('--n_valid', type=int, default=100) # using less examples so it's faster
#
parser.add_argument('--random_biases', type=int, default=1)

#parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'momentum', 'sgd'])
parser.add_argument('--save_dir', type=str, default="./")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--print_every', type=int, default=999999999)
parser.add_argument('--kl_weight', type=float, default=1.) # BUGGY!


# ---------------------------------------------------------------
# PARSE ARGS and SET-UP SAVING (save_path/exp_settings.txt)
# NTS: we name things after the filename + provided args.  We could also save versions (ala Janos), and/or time-stamp things.
# TODO: loading

args = parser.parse_args()
print args
args_dict = args.__dict__

if args_dict['save_dir'] is None:
    print "\n\n\n\t\t\t\t WARNING: save_dir is None! Results will not be saved! \n\n\n"
else:
    # save_dir = filename + PROVIDED parser arguments
    flags = [flag.lstrip('--') for flag in sys.argv[1:] if not flag.startswith('save_dir')]
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


# ---------------------------------------------------------------
# SET RANDOM SEED (TODO: rng vs. random.seed)

if seed is None:
    seed = np.random.randint(2**32 - 1)
np.random.seed(seed)  # for reproducibility
rng = np.random.RandomState(seed)
srng = RandomStreams(seed)
#set_rng(np.random.RandomState(seed))



# ---------------------------------------------------------------
# Get data
if os.path.isfile('/data/lisa/data/mnist.pkl.gz'):
    filename = '/data/lisa/data/mnist.pkl.gz'
elif os.path.isfile(r'./data/mnist.pkl.gz'):
    filename = r'./data/mnist.pkl.gz'
elif os.path.isfile(os.path.join(os.environ['DATA_PATH'], 'mnist.pkl.gz')):
    filename = os.path.join(os.environ['DATA_PATH'], 'mnist.pkl.gz')
else:        
    print '\n\tdownloading mnist'
    import download_datasets.mnist
    filename = r'./data/mnist.pkl.gz'

train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(filename)
train_x = train_x[:n_train]
train_y = train_y[:n_train]
va_x = valid_x[:n_valid]
va_y = valid_y[:n_valid]

len_split = n_train / n_splits
va_accs = np.zeros((n_splits, n_epochs-1))
# TODO: some analysis...
#va_means = np.zeros((n_splits, n_epochs))


# ---------------------------------------------------------------
# train model

prior_mean = 0
prior_log_var = 0

for split in range(n_splits):

    # get data
    tr_x = train_x[split*len_split:(split+1)*len_split]
    tr_y = train_y[split*len_split:(split+1)*len_split]

    # define model
    model = Full_BHN(
                 srng=srng,
                 prior_mean=prior_mean,
                 prior_log_var=prior_log_var,
                 n_hiddens=n_hiddens,
                 n_units=n_units,
                 #weight=kl_weight,
                 #dataset_size=len(tr_x),
                 random_biases=random_biases)

    model.X.tag.test_value = train_x[:32]
    model.y.tag.test_value = train_y[:32]

    # train and evaluate
    va_acc = train_model(model,tr_x,tr_y,va_x,va_y,
                lr0=lr,lrdecay=0,bs=bs,epochs=n_epochs,
                #anneal=0,name='0', e0=0,rec=0,
                save=0,
                verbose=verbose,
                kl_weight=kl_weight,
                print_every=print_every)

    va_accs[split] = va_acc

    # update posterior
    prior_mean = model.mean.eval()
    prior_log_var = model.log_var.eval()


np.save(save_dir + 'va_accs.npy', va_accs)










