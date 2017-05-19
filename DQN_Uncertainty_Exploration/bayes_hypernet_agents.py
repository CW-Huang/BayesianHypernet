import numpy as np

import theano
floatX = theano.config.floatX

class AgentEpsGreedy:

    def __init__(self, n_actions, value_function_model, state_dim, batch_size, eps=0.1):
        self.n_actions = n_actions
        self.value_func = value_function_model
        self.eps = eps
        self.state_dim = state_dim
        self.batch_size = batch_size


    def act(self, state):

        state = np.array([state])
        action_values = self.value_func.predict(state)

        policy = np.ones(self.n_actions) * self.eps / self.n_actions
        a_max = np.argmax(action_values)
        policy[a_max] += 1. - self.eps

        return np.random.choice(self.n_actions, p=policy)

    def thompson_hypernet_act(self, state):

        state = np.array([state])
        dropout_iterations=100
        dropout_acton_values = np.zeros(shape=(self.n_actions, dropout_iterations))

        for d in range(dropout_iterations):
            action_values = self.value_func.predict(state.astype(floatX))
            dropout_acton_values[:, d] = action_values

        mean_action_values = np.mean(dropout_acton_values, axis=1)

        return np.argmax(mean_action_values)


    def act_hypernet_EpsilonGreedy(self, state):

        state = np.array([state])

        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
            
        else:
            dropout_iterations=100
            dropout_acton_values = np.zeros(shape=(self.n_actions, dropout_iterations))

            for d in range(dropout_iterations):
                action_values = self.value_func.predict(state.astype(floatX))
                dropout_acton_values[:, d] = action_values

            mean_action_values = np.mean(dropout_acton_values, axis=1)

            return np.argmax(mean_action_values)



    def act_hypernet_epsilon_entropy(self, state):
        state = np.array([state])
        dropout_iterations=100

    
        if np.random.rand() < self.eps:

            dropout_acton_values_entropy = np.zeros(shape=(self.n_actions, dropout_iterations))
            
            for d in range(dropout_iterations):
                action_values_entropy = self.value_func.predict(state.astype(floatX))
                dropout_acton_values_entropy[:, d] = action_values_entropy

            mean_action_values_entropy = np.mean(dropout_acton_values_entropy, axis=1)
            log_mean =  np.log2(mean_action_values_entropy)

            Entropy_Average_Pi = - np.multiply(mean_action_values_entropy, log_mean)
            max_entropy_action = np.argmax(Entropy_Average_Pi)

            return max_entropy_action

        else:

            dropout_acton_values = np.zeros(shape=(self.n_actions, dropout_iterations))

            for d in range(dropout_iterations):
                action_values = self.value_func.predict(state.astype(floatX))
                dropout_acton_values[:, d] = action_values

            mean_action_values = np.mean(dropout_acton_values, axis=1)

            return np.argmax(mean_action_values)

    # TODO: more float problems here???
    def eval_train(self, states, targets, dropout_probability):
        return self.value_func.eval_train(states.astype(floatX), targets.astype(floatX), dropout_probability)


    def eval_valid(self, states, targets, dropout_probability):
        return self.value_func.eval_valid(states.astype(floatX), targets.astype(floatX), dropout_probability)


    def predict_q_values(self, states):
        return self.value_func.predict(states.astype(floatX))


    def evaluate_predicted_q_values(self, states, dropout_probability):
        return self.value_func.predict_stochastic(states.astype(floatX), dropout_probability)



    #good lr0 : 0.0001
    def train(self, X,Y, lr0=0.001,lrdecay=1,bs=20,epochs=50):
        X = X.astype(floatX)
        Y = Y.astype(floatX)
        

        train_func = self.value_func.train_func

        N = X.shape[0]    
        
        for e in range(epochs):
            
            if lrdecay:
                lr = lr0 * 10**(-e/float(epochs-1))
            else:
                lr = lr0         
                
            for i in range(N/bs):
                x = X[i*bs:(i+1)*bs]
                y = Y[i*bs:(i+1)*bs]
                
                loss = train_func(x,y,N,lr)
                
        return loss



    def act_boltzmann(self, state):
        action_values = self.value_func.predict([state])[0]
        action_values_tau = action_values / self.eps
        policy = np.exp(action_values_tau) / np.sum(np.exp(action_values_tau), axis=0)
        action_value_to_take = np.argmax(policy)
        return action_value_to_take










