"""
Implementation of Dropout based Thompson Sampling - estimating Q posterior at every state

and then taking the max_a (Q_mean) where Q_mean is the mean from the Q distribution 
"""

import copy
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import time
from utils import *
from ReplayMemory import ReplayMemory
from agents import AgentEpsGreedy

from valuefunctions import ValueFunctionDQN_TEST_TRAIN_DROPOUT
from lib import plotting


discount = 0.9
decay_eps = 0.9
batch_size = 64
max_n_ep = 1000    #originally defined! Don't change this.

min_avg_Rwd = 200000000  # Minimum average reward to consider the problem as solved
n_avg_ep = 100      # Number of consecutive episodes to calculate the average reward


def run_episode(env,
                agent,
                state_normalizer,
                memory,
                batch_size,
                discount,
                max_step=10000):
    state = env.reset()
    if state_normalizer is not None:
        state = state_normalizer.transform(state)[0]
    done = False
    total_reward = 0
    step_durations_s = np.zeros(shape=max_step, dtype=float)
    train_duration_s = np.zeros(shape=max_step-batch_size, dtype=float)
    progress_msg = "Step {:5d}/{:5d}. Avg step duration: {:3.1f} ms. Avg train duration: {:3.1f} ms. Loss = {:2.10f}."
    loss_v = 0
    w1_m = 0
    w2_m = 0
    w3_m = 0
    i = 0
    action = 0   


    #each step within an episode
    for i in range(max_step):
        t = time.time()
        if i > 0 and i % 200 == 0:
            print(progress_msg.format(i, max_step,
                                      np.mean(step_durations_s[0:i])*1000,
                                      np.mean(train_duration_s[0:i-batch_size])*1000,
                                      loss_v))
        if done:
            break

        """
        From dropout we have a posterior probability over the action value functions
        Choose action according to the posterior probability of being the best action
        """
        #actions based on maximising the q value
        action = agent.thompson_act(state)

        #take a, get s' and reward
        state_next, reward, done, info = env.step(action)
        total_reward += reward

        if state_normalizer is not None:
            state_next = state_normalizer.transform(state_next)[0]


        memory.add((state, action, reward, state_next, done))

        if len(memory.memory) > batch_size:  # DQN Experience Replay
            states_b, actions_b, rewards_b, states_n_b, done_b = zip(*memory.sample(batch_size))
            states_b = np.array(states_b)
            actions_b = np.array(actions_b)
            rewards_b = np.array(rewards_b)
            states_n_b = np.array(states_n_b)
            done_b = np.array(done_b).astype(int)


            """
            Prediction of Q values at evert state - action values on the arriving state
            """
            q_n_b = agent.predict_q_values(states_n_b)  # Action values on the arriving state

            #target - Q-learning here - taking max_a over Q(s', a)
            targets_b = rewards_b + (1. - done_b) * discount * np.amax(q_n_b, axis=1)

            #target function for the agent - predict based on the trained Q Network
            targets = agent.predict_q_values(states_b)
            for j, action in enumerate(actions_b):
                targets[j, action] = targets_b[j]

            t_train = time.time()

            #training the agent based on the target function
            loss_v, w1_m, w2_m, w3_m = agent.train(states_b, targets)
            train_duration_s[i - batch_size] = time.time() - t_train



        state = copy.copy(state_next)
        step_durations_s[i] = time.time() - t  # Time elapsed during this step
        step_length = time.time() - t



    return loss_v, w1_m, w2_m, w3_m, total_reward, step_length




env = gym.make("CartPole-v0")

n_actions = env.action_space.n
state_dim = env.observation_space.high.shape[0]


max_n_ep = 3000  #number of episodes
#max_step - number of steps within an episode

Experiments = 10
Experiments_All_Rewards = np.zeros(shape=(max_n_ep, Experiments))


for e in range(Experiments):


    #using a smaller network for the value function
    value_function = ValueFunctionDQN_TEST_TRAIN_DROPOUT(state_dim=state_dim, n_actions=n_actions, batch_size=batch_size)

    #this is the temperature parameter in Boltzmann Distribution
    tau_parameter = 1
    agent = AgentEpsGreedy(n_actions=n_actions, value_function_model=value_function, eps=tau_parameter)
    memory = ReplayMemory(max_size=100000)

    print('Experiment Number ', e)

    loss_per_ep = []
    w1_m_per_ep = []
    w2_m_per_ep = []
    w3_m_per_ep = []
    total_reward = []


    ep = 0
    avg_Rwd = -np.inf
    episode_end_msg = 'loss={:2.10f}, w1_m={:3.1f}, w2_m={:3.1f}, w3_m={:3.1f}, total reward={}'

    stats = plotting.EpisodeStats(episode_lengths=np.zeros(max_n_ep),episode_rewards=np.zeros(max_n_ep))  


    """
    Loop for the number of episodes : For every episode
    """
    while avg_Rwd < min_avg_Rwd and ep < max_n_ep:

        
        if ep >= n_avg_ep:
            avg_Rwd = np.mean(total_reward[ep-n_avg_ep:ep])
            print("EPISODE {}. Average reward over the last {} episodes: {}.".format(ep, n_avg_ep, avg_Rwd))
        else:
            print("EPISODE {}.".format(ep))


        """
        Contains loop for every step within the episode
        """
        loss_v, w1_m, w2_m, w3_m, cum_R, step_length = run_episode(env, agent, None, memory, batch_size=batch_size, discount=discount,
                                                      max_step=2000)
        

        print(episode_end_msg.format(loss_v, w1_m, w2_m, w3_m, cum_R))

        stats.episode_rewards[ep] = cum_R
        stats.episode_lengths[ep] = step_length


        #decaying parameter (epsilon in case of epsilon greedy, or temperature for Boltzmann)
        if agent.eps > 0.0001:
            agent.eps *= decay_eps

        # Collect episode results
        loss_per_ep.append(loss_v)
        w1_m_per_ep.append(w1_m)
        w2_m_per_ep.append(w2_m)
        w3_m_per_ep.append(w3_m)
        total_reward.append(cum_R)

        ep += 1



    Experiments_All_Rewards[:, e] = total_reward
    episode_length_over_time = stats.episode_lengths

    np.save('/home/ml/rislam4/Documents/BayesianHypernet/DQN_Uncertainty_Exploration/Non-Bayesian/Results'  + 'Thompson_CartPole' + '.npy', Experiments_All_Rewards)

env.close()


Average_Cum_Rwd = np.mean(Experiments_All_Rewards, axis=1)
np.save('/home/ml/rislam4/Documents/BayesianHypernet/DQN_Uncertainty_Exploration/Non-Bayesian/Results'  + 'Average_Thompson_CartPole' + '.npy', Average_Cum_Rwd)



