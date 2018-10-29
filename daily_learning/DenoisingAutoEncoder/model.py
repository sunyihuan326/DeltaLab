# coding:utf-8 
'''
created on 2018/4/25

@author:sunyihuan
'''
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data


def xaviver_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdadeltaOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.train_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(
            tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights["w1"]),
                   self.weights["b1"]))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights["w2"]), self.weights["b2"])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(self.reconstruction, self.x), 2.0)
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights["w1"] = tf.Variable(xaviver_init(self.n_input, self.n_hidden))
        all_weights["b1"] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights["w2"] = tf.Variable(xaviver_init(self.n_hidden, self.n_input))
        all_weights["b2"] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.train_scale})
        return cost

    def calc_total_fit(self, X):
        cost = self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.train_scale})
        return cost

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.train_scale})

    def gennerate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.train_scale})

    def getWeights(self):
        return self.sess.run(self.weights["w1"])

    def getBiases(self):
        return self.sess.run(self.weights["b1"])


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batchsize):
    start_index = np.random.randint(0, len(data) - batchsize)
    return data[start_index:(start_index + batchsize)]


X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
n_samples = int(mnist.train.num_examples)
training_epoches = 20
batch_size = 16
display_step = 1
autoencoder = AdditiveGaussianNoiseAutoEncoder(n_input=784, n_hidden=20, transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.01), scale=0.01)

for epoch in range(training_epoches):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batches_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batches_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:", "%04d" * (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
print("Total_cost:" + str(autoencoder.calc_total_fit(X_test)))
