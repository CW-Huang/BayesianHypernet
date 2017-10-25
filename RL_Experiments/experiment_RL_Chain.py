
import json
import numpy as np

#from keras.models_trial import Sequential
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from keras import backend as K
from keras.layers.core import Lambda

from dropout import MCdropout_MLP

#import theano
#theano.config.floatX='float32'

# model.predict
# model.train_on_batch

class Catch(object):
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.reset()

    def _update_state(self, action):
        """
        Input: action and states
        Ouput: new states and reward
        """
        state = self.state
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right
        f0, f1, basket = state[0]
        new_basket = min(max(1, basket + action), self.grid_size-1)
        f0 += 1
        out = np.asarray([f0, f1, new_basket])
        out = out[np.newaxis]

        assert len(out.shape) == 2
        self.state = out

    def _draw_state(self):
        im_size = (self.grid_size,)*2
        state = self.state[0]
        canvas = np.zeros(im_size)
        canvas[state[0], state[1]] = 1  # draw fruit
        canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
        return canvas

    def _get_reward(self):
        fruit_row, fruit_col, basket = self.state[0]
        if fruit_row == self.grid_size-1:
            if abs(fruit_col - basket) <= 1:
                return 1
            else:
                return -1
        else:
            return 0

    def _is_over(self):
        if self.state[0, 0] == self.grid_size-1:
            return True
        else:
            return False

    def observe(self):
        canvas = self._draw_state()
        return canvas.reshape((1, -1))

    def act(self, action):
        self._update_state(action)
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over


    def mc_dropout_act(self, model, input_tm):
        mc_samples = 50
        q_value = model.predict(input_tm)
        all_q_values = np.zeros(shape=(mc_samples, q_value.shape[1]))

        for m in range(mc_samples):
            q_value = model.predict(input_tm)
            all_q_values[m, :] = q_value

        mean_q_values = np.array([np.mean(all_q_values, axis=0)])
        action = np.argmax(mean_q_values[0])

        return action


    def reset(self):
        n = np.random.randint(0, self.grid_size-1, size=1)
        m = np.random.randint(1, self.grid_size-2, size=1)
        self.state = np.asarray([0, n, m])[np.newaxis]


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
    dropout=0.

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
    else:

        print ("MC Droput MLP")

        model = MCdropout_MLP(drop_prob=dropout,
                opt='momentum', # TODO: try others?
                              n_inputs=grid_size**2,
                              n_outputs=num_actions,
                              n_hiddens=2,
                              n_units=hidden_size,
                              output_type='real')
        model.predict = model.predict_proba
        # TODO: n=??
        model.train_on_batch = lambda x,y: model.train_func(x,y,n=1000, lr=.2)


    # If you want to continue training from a previous model, just uncomment the line bellow
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
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                action = env.mc_dropout_act(model, input_tm1)

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    if 0:
        # Save trained model weights and architecture, this will be used by the visualization code
        model.save_weights("model.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)