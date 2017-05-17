#!/usr/bin/env python
import numpy as np


class AgentEpsGreedy:
    def __init__(self, n_actions, value_function_model, eps=0.5):
        self.n_actions = n_actions
        self.value_func = value_function_model
        self.eps = eps

    def act(self, state):
        action_values = self.value_func.predict([state])[0]

        policy = np.ones(self.n_actions) * self.eps / self.n_actions
        a_max = np.argmax(action_values)
        policy[a_max] += 1. - self.eps

        return np.random.choice(self.n_actions, p=policy)

    def train(self, states, targets):
        return self.value_func.train(states, targets)

    def eval_train(self, states, targets, dropout_probability):
        return self.value_func.eval_train(states, targets, dropout_probability)

    # def eval_valid(self, states, targets, dropout_probability):
    #     return self.value_func.eval_valid(states, targets, dropout_probability)



    def eval_valid(self, states, targets, dropout_probability):
        return self.value_func.eval_valid(states, targets, dropout_probability)


    def predict_q_values(self, states):
        return self.value_func.predict(states)


    def evaluate_predicted_q_values(self, states, dropout_probability):
        return self.value_func.predict_stochastic(states, dropout_probability)



    # def predict_posterior_q_values(self, states):
    #     dropout_iterations = 100
    #     dropout_acton_values = np.array([[0, 0]]).T        
    #     for d in range(dropout_iterations):
    #         action_values = self.value_func.predict_posterior([states])[0]
    #         action_values = np.array([action_values]).T
    #         dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)
    #     dropout_acton_values = dropout_acton_values[:, 1:].T
    #     random_selection = np.random.randint(0,dropout_acton_values.shape[0],1)
    #     sampled_Q_value = dropout_acton_values[random_selection]
    #     return sampled_Q_value


    def dropout_highest_variance(self, state):

        '''
        Dropout probability is at 0.1 - predict function in valuefunctions.py
        Using MC Dropout - 5
        '''

        dropout_iterations = 100
        dropout_acton_values = np.array([[0, 0]]).T

        for d in range(dropout_iterations):
            action_values = self.value_func.predict_stochastic([state])[0]
            action_values = np.array([action_values]).T
            dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)

        dropout_acton_values = dropout_acton_values[:, 1:]
        #calculate variance across dropout estimates
        # variance over the action values
        variance = np.var(dropout_acton_values, 1)
        #pick action with highest variance over the action values        
        a_max = np.argmax(variance)      
        variance_value = np.max(variance)
        return a_max, variance_value






    def act_random(self, state):
        action = np.random.randint(0, self.n_actions)
        return action

    def thompson_act(self, state):
        action_values = self.value_func.predict([state])[0]
        a_max = np.argmax(action_values)
        return a_max


    def act_mc_dropout_Boltzmann(self, state):
        dropout_iterations = 100
        dropout_acton_values = np.array([[0, 0]]).T
        for d in range(dropout_iterations):
            action_values = self.value_func.predict_stochastic([state])[0]
            action_values = np.array([action_values]).T
            dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)
        dropout_acton_values = dropout_acton_values[:, 1:]
        mean_action_values = np.mean(dropout_acton_values, axis=1)
        Q_mean = mean_action_values 
        Q_mean_eps = np.divide(Q_mean, self.eps)
        Q_dist = np.exp(Q_mean_eps) / np.sum(np.exp(Q_mean_eps), axis=0)
        actions_to_take = np.argmax(Q_dist)
        return actions_to_take


    def act_boltzmann(self, state):
        action_values = self.value_func.predict([state])[0]
        action_values_tau = action_values / self.eps
        policy = np.exp(action_values_tau) / np.sum(np.exp(action_values_tau), axis=0)
        action_value_to_take = np.argmax(policy)
        return action_value_to_take



    def act_mc_dropout_EpsilonGreedy(self, state):

        dropout_iterations = 1
        dropout_acton_values = np.array([[0, 0]]).T

        for d in range(dropout_iterations):
            action_values = self.value_func.predict_stochastic([state])[0]
            action_values = np.array([action_values]).T
            dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)

        dropout_acton_values = dropout_acton_values[:, 1:]
        mean_action_values = np.mean(dropout_acton_values, axis=1)
        policy = np.ones(self.n_actions) * self.eps / self.n_actions
        a_max = np.argmax(mean_action_values)
        policy[a_max] += 1. - self.eps
        action = np.random.choice(self.n_actions, p=policy)

        return action



    def get_action_stochastic_Epsilon_Thompson_Sampling(self, state, drop_prob):

        dropout_iterations = 1
        dropout_acton_values = np.array([[0, 0]]).T

        for d in range(dropout_iterations):
            action_values = self.value_func.predict_stochastic([state])[0]
            action_values = np.array([action_values]).T
            dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)

        dropout_acton_values = dropout_acton_values[:, 1:]
        mean_action_values = np.mean(dropout_acton_values, axis=1)
        policy = np.ones(self.n_actions) * self.eps / self.n_actions
        a_max = np.argmax(mean_action_values)
        policy[a_max] += 1. - self.eps
        action = np.random.choice(self.n_actions, p=policy)

        return action




    def get_action_stochastic_epsilon_greedy(self, state, drop_prob):

        if np.random.rand() < self.eps:

            return np.random.randint(self.n_actions)

        else:

            dropout_iterations=1
            dropout_acton_values = np.array([range(self.n_actions)]).T

            for d in range(dropout_iterations):
                action_values = self.value_func.predict_stochastic([state], drop_prob)[0]
                action_values = np.array([action_values]).T
                dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)

            dropout_acton_values = dropout_acton_values[:, 1:]
            mean_action_values = np.mean(dropout_acton_values, axis=1)

            return np.argmax(mean_action_values)



    def get_action_stochastic(self, state, drop_prob):
        
        dropout_iterations=1
        dropout_action_values = np.array([range(self.n_actions)]).T

        for d in range(dropout_iterations):
            action_values = self.value_func.predict_stochastic([state], drop_prob)[0]
            action_values = np.array([action_values]).T
            dropout_action_values = np.append(dropout_action_values, action_values, axis=1)

        dropout_action_values = dropout_action_values[:, 1:]
        mean_action_values = np.mean(dropout_action_values, axis=1)

        return np.argmax(mean_action_values)









