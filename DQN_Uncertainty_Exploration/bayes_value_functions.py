import tensorflow as tf

import numpy as np


# 512 hidden units in layer 1, 256 units in layer2 - Larger Network
class ValueFunctionBayesHypernet:
    def __init__(self, state_dim=2, n_actions=3, batch_size=64):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Inputs, weights, biases and targets of the ANN
            self.x = tf.placeholder(tf.float32, shape=(None, state_dim))                    # Single sample
            self.train_data = tf.placeholder(tf.float32, shape=(batch_size, state_dim))     # Training batch of samples
            self.train_targets = tf.placeholder(tf.float32, shape=(batch_size, n_actions))  # Training batch of targets

            #self.l1_weights = tf.Variable(tf.truncated_normal([state_dim, 512], stddev=0.1), trainable=True, name="w1")
            self.l1_weights = tf.get_variable(name="w1", shape=[state_dim, 512],
                                              initializer=tf.contrib.layers.xavier_initializer())
            self.l1_biases = tf.Variable(tf.zeros([512]), trainable=True, name="b1")

            #self.l2_weights = tf.Variable(tf.truncated_normal([512, 256], stddev=0.1), trainable=True, name="w2")
            self.l2_weights = tf.get_variable(name="w2", shape=[512, 256],
                                              initializer=tf.contrib.layers.xavier_initializer())
            self.l2_biases = tf.Variable(tf.zeros([256]), trainable=True, name="b2")

            #self.l3_weights = tf.Variable(tf.truncated_normal([256, n_actions], stddev=0.1), trainable=True, name="w3")
            self.l3_weights = tf.get_variable(name="w3", shape=[256, n_actions],
                                              initializer=tf.contrib.layers.xavier_initializer())
            self.l3_biases = tf.Variable(tf.zeros([n_actions]), trainable=True, name="b3")

            # Interconnection of the various ANN nodes
            self.train_prediction = self.model(self.train_data)

            # Training calculations - squared difference between the TD Target and TD Predictions
            self.loss = tf.reduce_mean(tf.squared_difference(self.train_targets, self.train_prediction))

            #using Adam Optimizer for the training
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

            self.test_prediction = self.model(self.x)

            self.w1_max = tf.reduce_max(self.l1_weights)
            self.w2_max = tf.reduce_max(self.l2_weights)
            self.w3_max = tf.reduce_max(self.l3_weights)

            self.init_op = tf.global_variables_initializer()

        self.session = None

    def model(self, data):
        logits1 = tf.matmul(data, self.l1_weights) + self.l1_biases
        hidden1 = tf.nn.relu(logits1)  # Define units of layer

        logits2 = tf.matmul(hidden1, self.l2_weights) + self.l2_biases
        hidden2 = tf.nn.relu(logits2)  # Define units of layer

        return tf.matmul(hidden2, self.l3_weights) + self.l3_biases

    def init_tf_session(self):
        if self.session is None:
            self.session = tf.Session(graph=self.graph)
            self.session.run(self.init_op)  # Global Variables Initializer (init op)

    def predict(self, states):
        self.init_tf_session()  # Make sure the Tensorflow session exists

        feed_dict = {self.x: states}
        q = self.session.run(self.test_prediction, feed_dict=feed_dict)
        return q

    def train(self, states, targets):
        self.init_tf_session()  # Make sure the Tensorflow session exists

        feed_dict = {self.train_data: states, self.train_targets: targets}
        [l, _, w1_m, w2_m, w3_m] = self.session.run([self.loss, self.optimizer, self.w1_max, self.w2_max, self.w3_max],
                                                    feed_dict=feed_dict)

        return [l, w1_m, w2_m, w3_m]

    


