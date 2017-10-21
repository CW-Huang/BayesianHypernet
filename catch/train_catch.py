#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

from theano.tensor.shared_randomstreams import RandomStreams

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from keras import backend as K
from keras.layers.core import Lambda


from BHNs import MLPWeightNorm_BHN
from concrete_dropout import MLPConcreteDropout_BHN
#from utils import log_normal, log_laplace

from catch.mc_dropout_qlearn import Catch, ExperienceReplay, mc_greedy
from catch.dropout import MCdropout_MLP

# ---------------------------------------------------------------
import argparse
import os
import sys
import numpy 
np = numpy

parser = argparse.ArgumentParser()
parser.add_argument('--drop_prob', type=float, default=.0)
parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--exploration', type=str, default='epsilon_greedy', choices=['epsilon_greedy', 'RLSVI', 'TS'])
parser.add_argument('--lr', type=float, default=.2)
parser.add_argument('--model', type=str, default='MLE', choices=['MCdropout', 'MLE', 'BHN_WN'])
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_hids', type=int, default=100)
#
parser.add_argument('--opt', type=str, default='momentum', choices=['adam', 'momentum', 'sgd'])
parser.add_argument('--save_dir', type=str, default=None, help="save_dir must be set in order to save results!")
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--verbose', type=int, default=1)
#locals().update(parser.parse_args().__dict__)


# ---------------------------------------------------------------
# PARSE ARGS and SET-UP SAVING (save_dir/exp_settings.txt)
# NTS: we name things after the filename + provided args.  We could also save versions (ala Janos), and/or time-stamp things.
# TODO: loading

args = parser.parse_args()
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
lr = np.float32(lr)


# ---------------------------------------------------------------
# SET RANDOM SEED (TODO: rng vs. random.seed)


if seed is not None:
    np.random.seed(seed)  # for reproducibility
    rng = np.random.RandomState(seed)
else:
    rng = np.random.RandomState(np.random.randint(2**32 - 1))



# ---------------------------------------------------------------
# TODO: BELOW

# TODO: parameters
epsilon = .1  # exploration

grid_size = 10
n_actions = 3  # [move_left, stay, move_right]

max_memory = 500

batch_size = bs

hidden_size = n_hids
init_batch=None


# select model
if model == 'BHN_WN':
    model = MLPWeightNorm_BHN(lbda=1,
            opt=opt,
                              n_inputs=grid_size**2,
                              n_classes=n_actions,
                              srng = RandomStreams(seed=seed),
                              coupling=4,
                              n_hiddens=n_layers,
                              n_units=hidden_size,
                              flow='IAF',
                              output_type='real',
                              init_batch=init_batch)
elif model == 'MCdropout':
    model = MCdropout_MLP(drop_prob=drop_prob,
            opt=opt,
                              n_inputs=grid_size**2,
                              n_outputs=n_actions,
                              n_hiddens=n_layers,
                              n_units=hidden_size,
                              output_type='real')
elif model == 'MLE':
    model = MCdropout_MLP(drop_prob=0,
            opt=opt,
                              n_inputs=grid_size**2,
                              n_outputs=n_actions,
                              n_hiddens=n_layers,
                              n_units=hidden_size,
                              output_type='real')
else:
    assert False

# Define environment/game
env = Catch(grid_size)

# Initialize experience replay object
exp_replay = ExperienceReplay(max_memory=max_memory)

# Train
win_counts = []
win_count = 0
num_consecutive_wins = 0
best = 0
for e in range(n_epochs):
    loss = 0.
    env.reset()
    game_over = False

    # get initial input
    input_t = env.observe().astype("float32")

    # exploration policy
    if exploration == 'epsilon_greedy':
        def policy(state):
            if np.random.rand() <= epsilon:
                return  np.random.randint(0, n_actions)
            else:
                return  mc_greedy(model, state, n_mc_samples=20) # TODO: hardcoded
    elif exploration == 'RLSVI':
        # sample a q-network for the entire episode
        policy = lambda x: model.sample_qyx()(x).argmax()
    elif exploration == 'TS':
        # sample a q-network for each action
        policy = lambda state: mc_greedy(model, state, n_mc_samples=1)
    else:
        assert False

    # play one episode
    while not game_over:
        input_tm1 = (input_t).astype("float32")
        
        # get next action
        action = policy(input_tm1)
        #print action

        # apply action, get rewards and new state
        input_t, reward, game_over = env.act(action)
        if reward == 1:
            win_count += 1
            num_consecutive_wins += 1
        else:
            num_consecutive_wins = 0

        # save best
        if num_consecutive_wins > best:
            best = num_consecutive_wins
            if save_dir is not None:
                np.save(os.path.join(save_dir, 'win_counts.npy'), win_counts)
                model.save(os.path.join(save_dir, '.params'))

        # store experience
        exp_replay.remember([input_tm1, action, reward, input_t], game_over)

        # adapt model
        inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

        # FIXME: should be dataset size not batch size (?)
        loss += model.train_func(inputs.astype("float32"), targets.astype('float32'),batch_size, lr)

    win_counts.append(win_count)
    print("n_epochs {:03d}/{} | Loss {:.4f} | Win count {}".format(e, n_epochs, loss, win_count))


if save_dir is not None:
    np.save(os.path.join(save_dir, 'win_counts_final.npy'), win_counts)
    model.save(os.path.join(save_dir, '.params_final'))





