import numpy as np


class AgentEpsGreedy:
    def __init__(self, n_actions, value_function_model, eps=0.1):
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

    def eval_valid(self, states, targets, dropout_probability):
        return self.value_func.eval_valid(states, targets, dropout_probability)

    def predict_q_values(self, states):
        return self.value_func.predict(states)


    def evaluate_predicted_q_values(self, states, dropout_probability):
        return self.value_func.predict_stochastic(states, dropout_probability)


    def act_random(self, state):
        action = np.random.randint(0, self.n_actions)
        return action

    def thompson_act(self, state):
        action_values = self.value_func.predict([state])[0]
        a_max = np.argmax(action_values)
        return a_max



    def act_boltzmann(self, state):
        action_values = self.value_func.predict([state])[0]
        action_values_tau = action_values / self.eps
        policy = np.exp(action_values_tau) / np.sum(np.exp(action_values_tau), axis=0)
        action_value_to_take = np.argmax(policy)
        return action_value_to_take



    def dropout_thompson_act(self, state):

        drop_prob = 0.2
        dropout_iterations=100
        dropout_acton_values = np.array([range(self.n_actions)]).T

        for d in range(dropout_iterations):
            action_values = self.value_func.predict_stochastic([state], drop_prob)[0]
            action_values = np.array([action_values]).T
            dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)

        dropout_acton_values = dropout_acton_values[:, 1:]
        mean_action_values = np.mean(dropout_acton_values, axis=1)

        action = np.argmax(mean_action_values)

        return action


    def act_MCDropout_Epsilon_Greedy(self, state):

        drop_prob = 0.2
        if np.random.rand() < self.eps:

            return np.random.randint(self.n_actions)

        else:

            dropout_iterations = 100
            dropout_acton_values = np.zeros(shape=(self.n_actions, dropout_iterations))
            
            for d in range(dropout_iterations):
                action_values = self.value_func.predict_stochastic([state], drop_prob)[0]
                dropout_acton_values[:, d] = action_values

            mean_action_values = np.mean(dropout_acton_values, axis=1)

            return np.argmax(mean_action_values)



    def act_dropout_epsilon_entropy(self, state):
        drop_prob = 0.2
        dropout_iterations=100
        if np.random.rand() < self.eps:

            dropout_acton_values_entropy = np.zeros(shape=(self.n_actions, dropout_iterations))
            
            for d in range(dropout_iterations):
                action_values_entropy = self.value_func.predict_stochastic([state], drop_prob)
                dropout_acton_values_entropy[:, d] = action_values_entropy

            mean_action_values_entropy = np.mean(dropout_acton_values_entropy, axis=1)
            log_mean =  np.log2(mean_action_values_entropy)

            Entropy_Average_Pi = - np.multiply(mean_action_values_entropy, log_mean)
            max_entropy_action = np.argmax(Entropy_Average_Pi)

            return max_entropy_action


        else:

            dropout_acton_values = np.zeros(shape=(self.n_actions, dropout_iterations))

            for d in range(dropout_iterations):
                action_values = self.value_func.predict_stochastic([state], drop_prob)[0]
                action_values = np.array([action_values]).T
                dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)

            dropout_acton_values = dropout_acton_values[:, 1:]
            mean_action_values = np.mean(dropout_acton_values, axis=1)

            return np.argmax(mean_action_values)


    def epsilon_max_entropy(self, state):


        dropout_iterations=100
        if np.random.rand() < self.eps:

            dropout_acton_values_entropy = np.zeros(shape=(self.n_actions, dropout_iterations))
            
            for d in range(dropout_iterations):
                action_values_entropy = self.value_func.predict([state])
                dropout_acton_values_entropy[:, d] = action_values_entropy

            mean_action_values_entropy = np.mean(dropout_acton_values_entropy, axis=1)
            log_mean =  np.log2(mean_action_values_entropy)

            Entropy_Average_Pi = - np.multiply(mean_action_values_entropy, log_mean)
            max_entropy_action = np.argmax(Entropy_Average_Pi)

            return max_entropy_action


        else:


            dropout_acton_values = np.array([range(self.n_actions)]).T

            for d in range(dropout_iterations):
                action_values = self.value_func.predict([state])[0]
                action_values = np.array([action_values]).T
                dropout_acton_values = np.append(dropout_acton_values, action_values, axis=1)

            dropout_acton_values = dropout_acton_values[:, 1:]
            mean_action_values = np.mean(dropout_acton_values, axis=1)

            return np.argmax(mean_action_values)











