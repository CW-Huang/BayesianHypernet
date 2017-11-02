#!/usr/bin/env python

"""
WIP

This script implements the idea of sequential Bayesian updating for BDNNs with VI approximate posteriors.
"""

from BHNs import Base_BHN
import modules
from modules import LinearFlowLayer, IndexLayer#, PermuteLayer, SplitLayer, ReverseLayer
from ops import load_mnist
from utils import log_normal, log_laplace, train_model, evaluate_model

import lasagne
from lasagne import nonlinearities
from lasagne.layers import get_output
from lasagne.random import set_rng

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
floatX = theano.config.floatX

# From BHNs.py
from modules import LinearFlowLayer, IndexLayer, PermuteLayer, SplitLayer, ReverseLayer
from modules import CoupledDenseLayer, ConvexBiasLayer, CoupledWNDenseLayer, \
                    stochasticDenseLayer2, stochasticConv2DLayer, \
                    stochastic_weight_norm
from modules import *
from utils import log_normal
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
RSSV = T.shared_randomstreams.RandomStateSharedVariable
floatX = theano.config.floatX

import lasagne
from lasagne import nonlinearities
rectify = nonlinearities.rectify
softmax = nonlinearities.softmax
from lasagne.layers import get_output
from lasagne.objectives import categorical_crossentropy as cc
from lasagne.objectives import squared_error as se
import numpy as np

from helpers import flatten_list
from helpers import SaveLoadMIXIN

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
    return T.log(prior_var) - T.log(posterior_var) + (posterior_var**2 + (prior_mean - posterior_mean)**2) / 2. / prior_var**2  - .5


class Full_BHN(Base_BHN):
    """
    hypernet (really just BbB/mean field for now) that outputs ALL the primary net parameters (including biases!!)
    """
    def __init__(self,
                 lbda=1,
                 perdatapoint=False,
                 srng = RandomStreams(seed=427),
                 #prior = log_normal,
                 prior_mean = 0,
                 prior_log_var = 0,
                 coupling=0,
                 n_hiddens=2,
                 n_units=800,
                 n_inputs=784,
                 n_classes=10,
                 output_type = 'categorical',
                 random_biases=1,
                 **kargs):
        
        self.__dict__.update(locals())
        assert coupling == 0# TODO

        self.weight_shapes = list()        
        if n_hiddens > 0:
            self.weight_shapes.append((n_inputs,n_units))
            for i in range(1,n_hiddens):
                self.weight_shapes.append((n_units,n_units))
            self.weight_shapes.append((n_units,n_classes))
        else:
            self.weight_shapes = [(n_inputs, n_classes)]

        if self.random_biases:
            self.num_params = sum((ws[0]+1)*ws[1] for ws in self.weight_shapes)
        else:
            self.num_params = sum((ws[0])*ws[1] for ws in self.weight_shapes)
        
        super(Full_BHN, self).__init__(lbda=lbda,
                                                perdatapoint=perdatapoint,
                                                srng=srng,
                                                prior=None,
                                                output_type = output_type,
                                                **kargs)
        assert self.wd1 == 1
    
    def _get_hyper_net(self):
        # inition random noise
        self.ep = self.srng.normal(size=(self.wd1,
                                    self.num_params),dtype=floatX)
        logdets_layers = []
        h_net = lasagne.layers.InputLayer([None,self.num_params])
        
        # mean and variation of the initial noise
        layer_temp = LinearFlowLayer(h_net)
        self.mean = layer_temp.b
        self.log_var = layer_temp.W
        self.delta = .001 # default value from modules.py
        self.h_net = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))

        self.weights = lasagne.layers.get_output(h_net,self.ep)
        self.logdets = sum([get_output(ld,self.ep) for ld in logdets_layers])
    
    def _get_primary_net(self):
        t = 0#np.cast['int32'](0) # TODO: what's wrong with np.cast
        p_net = lasagne.layers.InputLayer([None,self.n_inputs])
        inputs = {p_net:self.input_var}
        for ws in self.weight_shapes:
            if self.random_biases:
                num_param = (ws[0]+1) * ws[1]
                weight_and_bias = self.weights[:,t:t+num_param]#.reshape((self.wd1,ws[0], ws[1]))
                weight = weight_and_bias[:,:ws[0]*ws[1]].reshape((self.wd1, ws[0], ws[1]))
                w_layer = lasagne.layers.InputLayer((None,ws[0]*ws[1]))
                inputs[w_layer] = weight
                bias = weight_and_bias[:,ws[0]*ws[1]:].reshape((self.wd1, ws[1]))
                b_layer = lasagne.layers.InputLayer((None,ws[1]))
                inputs[b_layer] = bias
                p_net = modules.stochasticDenseLayerWithBias([p_net, w_layer, b_layer], num_units=ws[1])
            else:
                num_param = (ws[0]) * ws[1]
                weight_and_bias = self.weights[:,t:t+num_param]#.reshape((self.wd1,ws[0], ws[1]))
                weight = weight_and_bias[:,:ws[0]*ws[1]].reshape((self.wd1, ws[0], ws[1]))
                w_layer = lasagne.layers.InputLayer((None,ws[0]*ws[1]))
                inputs[w_layer] = weight
                #bias = weight_and_bias[:,ws[0]*ws[1]:].reshape((self.wd1, ws[1]))
                #b_layer = lasagne.layers.InputLayer((None,ws[1]))
                #inputs[b_layer] = bias
                p_net = modules.stochasticDenseLayer([p_net, w_layer], num_units=ws[1])

            #print p_net.output_shape
            t += num_param
            
        if self.output_type == 'categorical':
            p_net.nonlinearity = nonlinearities.softmax
            y = T.clip(get_output(p_net,inputs), 0.001, 0.999) # stability
            self.p_net = p_net
            self.y = y
            self.y_unclipped = get_output(p_net,inputs)
        elif self.output_type == 'real':
            p_net.nonlinearity = nonlinearities.linear
            y = get_output(p_net,inputs)
            self.p_net = p_net
            self.y = y
            self.y_unclipped = get_output(p_net,inputs)
        else:
            assert False

    # TODO
    def _get_elbo(self):
        self.kl = KL(self.prior_mean, self.prior_log_var,
                     self.mean, self.log_var).sum(-1).mean()

        if self.output_type == 'categorical':
            self.logpyx = - cc(self.y,self.target_var).mean()
        elif self.output_type == 'real':
            self.logpyx = - se(self.y,self.target_var).mean()
        else:
            assert False
        self.loss = - (self.logpyx - \
                       self.weight * self.kl/T.cast(self.dataset_size,floatX))

        # DK - extra monitoring
        params = self.params
        ds = self.dataset_size
        self.logpyx_grad = flatten_list(T.grad(-self.logpyx, params, disconnected_inputs='warn')).norm(2)
        self.monitored = [self.logpyx, self.kl]

        
    def _get_useful_funcs(self):
        self.predict_proba = theano.function([self.input_var],self.y)
        self.predict = theano.function([self.input_var],self.y.argmax(1))
        self.predict_fixed_mask = theano.function([self.input_var, self.weights],self.y)
        self.sample_weights = theano.function([], self.weights)
    

# ---------------------------------------------------------------
import argparse
import os
import sys
import numpy 
np = numpy

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--lr', type=float, default=.001)
#
parser.add_argument('--n_epochs', type=int, default=11)
parser.add_argument('--n_hiddens', type=int, default=1)
parser.add_argument('--n_units', type=int, default=200)
parser.add_argument('--n_splits', type=int, default=1)
parser.add_argument('--n_train', type=int, default=5000)
parser.add_argument('--n_valid', type=int, default=1000) # using less examples so it's faster
#
parser.add_argument('--random_biases', type=int, default=1)

#parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'momentum', 'sgd'])
parser.add_argument('--save_dir', type=str, default="./")
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--verbose', type=int, default=1)


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
set_rng(np.random.RandomState(seed))



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
va_accs = np.zeros((n_splits, n_epochs))
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
                 random_biases=random_biases)

    model.input_var.tag.test_value = train_x[:32]
    model.target_var.tag.test_value = train_x[:32]

    # train and evaluate
    va_acc = train_model(model,tr_x,tr_y,va_x,va_y,
                lr0=lr,lrdecay=0,bs=bs,epochs=n_epochs,
                #anneal=0,name='0', e0=0,rec=0,
                save=0,
                verbose=verbose,
                print_every=9999999999)

    va_accs[split] = va_acc

    # update posterior
    prior_mean = model.mean.eval()
    prior_log_var = model.log_var.eval()


np.save(save_dir + 'va_accs.npy', va_accs)










