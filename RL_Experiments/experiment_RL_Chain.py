from dropout import MCdropout_MLP
from BHNs import MLPWeightNorm_BHN
from theano.tensor.shared_randomstreams import RandomStreams
import argparse
import os
import sys
import numpy 
np = numpy
seed = np.random.randint(2**32-1)

parser = argparse.ArgumentParser()
parser.add_argument('--drop_prob', type=float, default=.0)
#parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--exploration', type=str, default='epsilon_greedy', choices=['epsilon_greedy', 'RLSVI', 'TS'])
parser.add_argument('--lr', type=float, default=.2)
parser.add_argument('--model', type=str, default='BHN', choices=['BHN', 'MCD'])
#parser.add_argument('--n_epochs', type=int, default=1000)
#parser.add_argument('--n_layers', type=int, default=2)
#parser.add_argument('--n_hids', type=int, default=100)
#
#parser.add_argument('--opt', type=str, default='momentum', choices=['adam', 'momentum', 'sgd'])
#parser.add_argument('--save_dir', type=str, default=None, help="save_dir must be set in order to save results!")
#parser.add_argument('--seed', type=int, default=1337)
#parser.add_argument('--verbose', type=int, default=1)
locals().update(parser.parse_args().__dict__)



class Chain(object):
    def __init__(self, nState, nAction, epLen): 
        '''
        Initialize a tabular episodic MDP

        Args:
            nState  - int - number of states
            nAction - int - number of actions
            epLen   - int - episode length

        Returns:
            Environment object
        '''

        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen

        self.timestep = 0
        self.state = 0

        # Now initialize R and P
        self.R = {}
        self.P = {}
        for state in range(nState):
            for action in range(nAction):
                self.R[state, action] = (1, 1)
                self.P[state, action] = np.ones(nState) / nState


    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = 0
          

    def act(self, action):
        if self.R[self.state, action][1] < 1e-9:
            # Hack for no noise
            reward = self.R[self.state, action][0]
        else:
            reward = np.random.normal(loc=self.R[self.state, action][0],
                                      scale=self.R[self.state, action][1])
        newState = np.random.choice(self.nState, p=self.P[self.state, action])

        # Update the environment
        self.state = newState
        self.timestep += 1

        if self.timestep == self.epLen:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1

        return reward, newState, pContinue



    def observe(self):
        self.state = np.array([0])
        return self.state


def make_bootDQNChain(nState=6, epLen=15, nAction=2):
    '''
    Creates the chain from Bootstrapped DQN

    Returns:
        bootDQNChain - Tabular MDP environment
    '''
    R_true = {}
    P_true = {}

    for s in xrange(nState):
        for a in xrange(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (0.01, 1)
    R_true[nState - 1, 1] = (1, 1)

    # Transitions
    for s in xrange(nState):
        P_true[s, 0][max(0, s-1)] = 1.

        P_true[s, 1][min(nState - 1, s + 1)] = 0.5
        P_true[s, 1][max(0, s-1)] = 0.5

    bootDQNChain = Chain(nState, nAction, epLen)
    bootDQNChain.R = R_true
    bootDQNChain.P = P_true
    bootDQNChain.reset()

    return bootDQNChain





class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = 3#model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


if __name__ == "__main__":
    # parameters
    epsilon = .1  # exploration
    num_actions = 3  # [move_left, stay, move_right]
    epoch = 1000
    max_memory = 500
    hidden_size = 100
    batch_size = 50
    grid_size = 10
    dropout=0.1

    exploration = 'RLSVI'
    #exploration = 'TS'
    #exploration = 'epsilon_greedy'

    if 0:
        model = Sequential()
        model.add(Dense(hidden_size, input_shape=(grid_size**2,), activation='relu'))
        if dropout:
            model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
        model.add(Dense(hidden_size, activation='relu'))
        if dropout:
            model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
        model.add(Dense(num_actions))
        if dropout: # TODO: rm this! (no dropout at the output, brah!
            model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
        model.compile(sgd(lr=.2), "mse")
    elif model == 'MCD':
        model = MCdropout_MLP(drop_prob=dropout,
                opt='momentum', # TODO: try others?
                              n_inputs=grid_size**2,
                              n_outputs=num_actions,
                              n_hiddens=2,
                              n_units=hidden_size,
                              output_type='real')
        model.predict = model.predict_proba
        # FIXME: n=??
        model.train_on_batch = lambda x,y: model.train_func(x,y,n=1000, lr=lr)
    elif model == 'BHN':
        model = MLPWeightNorm_BHN(lbda=10.,
                opt='momentum',
                                  n_inputs=grid_size**2,
                                  n_classes=num_actions,
                                  srng = RandomStreams(seed=seed),
                                  coupling=0,
                                  n_hiddens=2,
                                  n_units=hidden_size,
                                  flow='IAF',
                                  output_type='real')
        model.predict = model.predict_proba
        # FIXME: n=??
        model.train_on_batch = lambda x,y: model.train_func(x,y,n=1000, lr=lr)
    else: 
        assert False



    # Define environment/game
    # env = Catch(grid_size)

    env = make_bootDQNChain()

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        # TODO: TESTING    
        if exploration == 'epsilon_greedy':
            def policy(state):
                if np.random.rand() <= epsilon:
                    return  np.random.randint(0, num_actions)
                else:
                    return  env.mc_dropout_act(model, state, mc_samples=50) # TODO: hardcoded
        elif exploration == 'RLSVI':
            # sample a q-network for the entire episode
            policy = lambda x: model.sample_qyx()(x).argmax()
        elif exploration == 'TS':
            # sample a q-network for each action
            policy = lambda state: env.mc_dropout_act(model, state, mc_samples=1)
        else:
            assert False


        ## eplen : timesteps in the environment (max t=15)
        eplen = 15
        for e in range(eplen):

        # while not game_over:
            input_tm1 = input_t.astype("float32")
            # get next action
            action = policy(input_tm1)

            import pdb; pdb.set_trace()

            if 0:
                if np.random.rand() <= epsilon:
                    action = np.random.randint(0, num_actions, size=1)
                else:
                    action = env.mc_dropout_act(model, input_tm1)

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            input_t = input_t.astype("float32")#, reward, game_over = env.act(action)
             #= input_t.astype("float32")#, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs.astype("float32"), targets.astype("float32"))
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    if 0:
        # Save trained model weights and architecture, this will be used by the visualization code
        model.save_weights("model.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)

