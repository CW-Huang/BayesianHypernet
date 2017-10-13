import json

from theano.tensor.shared_randomstreams import RandomStreams

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from keras import backend as K
from keras.layers.core import Lambda

from mc_dropout_qlearn import Catch, ExperienceReplay

from BHNs import MLPWeightNorm_BHN
from concrete_dropout import MLPConcreteDropout_BHN
#from utils import log_normal, log_laplace

from dropout import MCdropout_MLP

# ---------------------------------------------------------------
import argparse
import os
import sys
import numpy 
np = numpy

parser = argparse.ArgumentParser()
parser.add_argument('--exploration', type=str, default='RLSVI', choices=['epsilon_greedy', 'RLSVI', 'TS'])
parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--model', type=str, default='BHN_WN', choices=['MCdropout', 'MLE', 'BHN_WN'])
#
#parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'momentum', 'sgd'])
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
num_actions = 3  # [move_left, stay, move_right]

epoch = 1000
max_memory = 500

batch_size = 50

hidden_size = 100
num_layers = 3
init_batch=None


# TODO: optimizer
if model == 'BHN_WN':
    framework = 'lasagne'
    model = MLPWeightNorm_BHN(lbda=1,
                              n_inputs=grid_size**2,
                              n_classes=num_actions,
                              srng = RandomStreams(seed=args.seed+2000),
                              coupling=4,
                              n_hiddens=num_layers,
                              n_units=hidden_size,
                              flow='IAF',
                              output_type='real',
                              init_batch=init_batch)
elif model == 'MCdropout':
    framework = 'lasagne'
    model = MCdropout_MLP(
                              n_inputs=grid_size**2,
                              n_outputs=num_actions,
                              n_hiddens=num_layers,
                              n_units=hidden_size,
                              output_type='real')
# TODO:
elif model == 'MLE':
    framework = 'keras'
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=lr), "mse")
else:
    assert False


# TODO: If you want to continue training from a previous model, just uncomment the line bellow
# model.load_weights("model.h5")

# Define environment/game
env = Catch(grid_size)

# Initialize experience replay object
exp_replay = ExperienceReplay(max_memory=max_memory)

# Train
win_cnt = 0
for e in range(epoch):
    loss = 0.
    env.reset()
    game_over = False
    # get initial input
    input_t = env.observe().astype("float32")

    if exploration == 'epsilon_greedy':
        def policy(state):
            if np.random.rand() <= epsilon:
                return  np.random.randint(0, num_actions, size=1)
            else:
                return  mc_greedy(model, state, num_mc_samples=20) # TODO: hardcoded
    elif exploration == 'RLSVI':
        # sample a q-network for the entire episode
        policy = lambda x: model.sample_qyx()(x).argmax()
    elif exploration == 'TS':
        # sample a q-network for each action
        policy = lambda state: mc_greedy(model, state, num_mc_samples=1)
    else:
        assert False

    while not game_over:
        input_tm1 = (input_t).astype("float32")
        
        # get next action
        action = policy(input_tm1)

        # apply action, get rewards and new state
        input_t, reward, game_over = env.act(action)
        if reward == 1:
            win_cnt += 1

        # store experience
        exp_replay.remember([input_tm1, action, reward, input_t], game_over)

        # adapt model
        inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

        loss += model.train_func(inputs.astype("float32"), targets.astype('float32'),batch_size, lr)
    print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))


assert False # TODO


if framework == 'keras':
    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
elif framework == 'lasagne':
    pass # TODO: saving!
with open("model.json", "w") as outfile:
    json.dump(model.to_json(), outfile)




