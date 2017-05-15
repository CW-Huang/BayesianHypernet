# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:46:38 2017

@author: Chin-Wei (and David Krueger :D)

TODO:
    implement baselines... 

    logging
    double-check math
    launchable-ness (SLURM jobs)
    test-time MC (THEANO)



"""


# TODO: make it work!!!
#   am I doing it wrong????
# what is different??
"""
initialization
optimization
features (binary vs. categorical)
experience sampling

"""


from modules import LinearFlowLayer, IndexLayer, PermuteLayer
from modules import CoupledDenseLayer, CoupledConv1DLayer
from utils import log_stdnormal
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



class stochasticDenseLayer(lasagne.layers.base.MergeLayer):
    
    def __init__(self, incomings, num_units, 
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 num_leading_axes=1, **kwargs):
        super(stochasticDenseLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)
        self.num_units = num_units
        
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)
    def get_output_shape_for(self,input_shapes):
        input_shape = input_shapes[0]
        weight_shape = input_shapes[1]
        return (input_shape[0], weight_shape[2])
        
    def get_output_for(self, inputs, **kwargs):
        """
        inputs[0].shape = (None, num_inputs)
        inputs[1].shape = (None/1, num_inputs, num_units)
        """
        input = inputs[0]
        W = inputs[1]
        activation = T.sum(input.dimshuffle(0,1,'x') * W, axis = 1)
        if self.b is not None:
            activation = activation + self.b
        return self.nonlinearity(activation)



# TODO: use init_W instead of std!
# TODO: prior?

if 1:
#def main():
    """
    MNIST example
    """
    
    import argparse
    
    parser = argparse.ArgumentParser()
    # different noise for each datapoint (vs. each minibatch)
    parser.add_argument('--bs',default=64,type=int)  
    parser.add_argument('--burn_in',default=100,type=int)  
    parser.add_argument('--coupling',default=0,type=bool)  
    parser.add_argument('--epochs',default=50000,type=int)  
    parser.add_argument('--init_lr',default=.1,type=float)  
    parser.add_argument('--mem_size',default=4096,type=int)  
    parser.add_argument('--num_hids',default=100,type=int)  
    parser.add_argument('--opt',default='momentum',type=str)  
    parser.add_argument('--perdatapoint',default=0,type=bool)    
    parser.add_argument('--std',default=1.,type=bool)    
    # num_ex
    parser.add_argument('--size',default=10000,type=int)  
    args = parser.parse_args()
    locals().update(args.__dict__)
    print args

    perdatapoint = args.perdatapoint
    coupling = args.coupling
    size = max(10,min(50000,args.size))
    print "size",size
    # these seem large!
    clip_grad = 100
    max_norm = 1000
    
    # load dataset
    from mushroom_data import X,Y 
    
    # THEANO VARS
    input_var = T.matrix('input_var')
    input_var.tag.test_value = np.random.randn(bs, 119).astype('float32')
    #action_var = T.matrix('input_var')
    target_var = T.matrix('target_var')
    target_var.tag.test_value = np.random.randn(bs, 1).astype('float32')
    dataset_size = T.scalar('dataset_size')
    dataset_size.tag.test_value = np.float32(bs)
    lr = T.scalar('lr') 
    lr.tag.test_value = np.float32(1.)
    
    weight_shapes = [(119, num_hids),
                     (num_hids, num_hids),
                     (num_hids,  1)]

    num_params = sum(np.prod(ws) for ws in weight_shapes)
    if perdatapoint:
        wd1 = input_var.shape[0]
    else:
        wd1 = 1

    # stochastic hypernet    
    ep = srng.normal(size=(wd1,num_params), std=std, dtype=floatX)
    logdets_layers = []
    h_layer = lasagne.layers.InputLayer([None,num_params])
    
    layer_temp = LinearFlowLayer(h_layer)
    h_layer = IndexLayer(layer_temp,0)
    logdets_layers.append(IndexLayer(layer_temp,1))
    
    if coupling:
        layer_temp = CoupledConv1DLayer(h_layer,16,5)
        h_layer = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
        
        h_layer = PermuteLayer(h_layer,num_params)
        
        layer_temp = CoupledConv1DLayer(h_layer,16,5)
        h_layer = IndexLayer(layer_temp,0)
        logdets_layers.append(IndexLayer(layer_temp,1))
    
    # pseudo-params
    weights = lasagne.layers.get_output(h_layer,ep)
    
    # primary net
    t = np.cast['int32'](0)
    layer = lasagne.layers.InputLayer([None,784])
    inputs = {layer:input_var}
    for ws in weight_shapes: # TODO: perdatapoint will break here!
        num_param = np.prod(ws)
        #print t, t+num_param
        w_layer = lasagne.layers.InputLayer((None,)+ws)
        weight = weights[:,t:t+num_param].reshape((wd1,)+ws)
        inputs[w_layer] = weight
        layer = stochasticDenseLayer([layer,w_layer],ws[1])
        t += num_param
        
    #layer.nonlinearity = nonlinearities.softmax
    y = get_output(layer,inputs)
    #y = T.clip(y, 0.00001, 0.99999) # stability 
    
    # loss terms
    # TODO: monitor separately
    logdets = sum([get_output(logdet,ep) for logdet in logdets_layers])
    # FIXME: are we using *different* epsilons when we should be using the same ones???
    logqw = - (0.5*(ep**2).sum(1) + 0.5*T.log(2*np.pi)*num_params + logdets)
    logpw = log_stdnormal(weights).sum(1)
    kl = (logqw - logpw).mean()
    logpyx = - ((y - target_var)**2).mean()
    loss = - (logpyx - kl/T.cast(dataset_size,floatX))
    outputs = [loss, logpyx, logdets]
    
    params = lasagne.layers.get_all_params([h_layer,layer])
    grads = T.grad(loss, params)
    # double clipping??
    mgrads = lasagne.updates.total_norm_constraint(grads,
                                                   max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]
    if opt == 'adam':
        updates = lasagne.updates.adam(cgrads, params, learning_rate=lr)
    elif opt == 'momentum':
        updates = lasagne.updates.nesterov_momentum(cgrads, params, learning_rate=lr)
    
    # THEANO FUNCTIONS
    train = theano.function([input_var,target_var,dataset_size,lr],
                            outputs,
                            updates=updates)
    predict = theano.function([input_var],y)
    
    ###########################
    # TRAIN MODEL

    # actions:
    eat = np.array([1,0]).astype('float32')
    dont = np.array([0,1]).astype('float32')

    # experience replay (TODO: these should really be arrays....)
    context_buffer = np.empty((mem_size, 117)).astype('float32')
    action_buffer = np.empty((mem_size, 2)).astype('float32')
    reward_buffer = np.empty((mem_size, 1)).astype('float32')

    # logging 
    cum_regret = 0
    cum_regrets = np.zeros(epochs)

    for interaction in range(epochs):
        memory_ind = interaction % mem_size
        
        print "interaction", interaction
        
        # train (TODO: start training after 64 interactions??)
        if interaction > burn_in:
            for _ in range(64):
                # TODO
                batch = np.random.choice(range(min(mem_size, interaction)), bs) # replacement?
                contexts = context_buffer[batch]
                actions = action_buffer[batch]
                rewards = reward_buffer[batch]
                inputs = np.hstack((contexts, actions))
                train(inputs, rewards, bs, init_lr)

        # sample context
        ind = np.random.choice(range(len(X)))
        context = X[[ind]]
        is_edible = Y[ind]

        # predict reward
        expected_reward = [np.mean([predict(np.hstack((context, eat.reshape((1, 2))))) for sample in range(2)]),
                           np.mean([predict(np.hstack((context, dont.reshape((1, 2))))) for sample in range(2)])]

        # select action
        if expected_reward[0] > expected_reward[1]:
            action = eat
        else:
            action = dont

        # take action and collect reward
        if is_edible:
            if np.all(action == eat):
                reward = 5.
                cum_regrets[interaction] = cum_regret
            else:
                reward = 0.
                cum_regret += 5.
                cum_regrets[interaction] = cum_regret
        else: # POISON!!!!!!!!!!!!!!!!!!!!
            if np.all(action == eat):
                reward = 5. - 40 * (np.random.rand() > .5)
                cum_regret += 15.
                cum_regrets[interaction] = cum_regret
            else:
                reward = 0.
                cum_regrets[interaction] = cum_regret

        # record experience
        context_buffer[memory_ind] = context
        action_buffer[memory_ind] = action
        reward_buffer[memory_ind] = reward

    # END TRAIN MODEL
    ###########################


