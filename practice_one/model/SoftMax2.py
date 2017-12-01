# coding:utf-8
'''
Created on 2017/11/15.

@author: chk01
'''

# 读取数据
# 数据预处理-reshape-标准化
# 每一步迭代步骤
# 循环迭代步骤
import os
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import math


def init_sets(X, Y, file, distribute):
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]
    assert len(distribute) == 2
    assert sum(distribute) == 1
    scio.savemat(file + '_train',
                 {'X': shuffled_X[:int(m * distribute[0]), :], 'Y': shuffled_Y[:int(m * distribute[0]), :]})
    scio.savemat(file + '_test',
                 {'X': shuffled_X[int(m * distribute[0]):, :], 'Y': shuffled_Y[int(m * distribute[0]):, :]})
    return True


def load_data(file):
    if not os.path.exists(file + '_test.mat'):
        data = scio.loadmat(file)
        m = data['X'].shape[0]
        x = data['X'].reshape(m, -1)
        y = np.squeeze(data['Y']).T
        print(x.shape)
        print(y.shape)
        init_sets(x, y, file, distribute=[0.8, 0.2])
    # (1361, 64, 64, 3)
    # (1361, 9)
    return True


def initialize_parameters(n_x, n_y, file, ifExtend=False):
    W1 = tf.get_variable(name='W1', dtype=tf.float32, shape=(n_y, n_x),
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.constant(0.1)
    if ifExtend and os.path.exists(file + '_parameters'):
        parameters = scio.loadmat(file + '_parameters')
        W1 = tf.Variable(parameters['W1'])
        b1 = tf.Variable(parameters['b1'])

    parameters = {"W1": W1, 'b1': b1}

    return parameters


def create_placeholders(n_x, n_y):
    X = tf.placeholder(name='X', shape=(None, n_x), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(None, n_y), dtype=tf.float32)

    return X, Y


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']

    Z1 = tf.add(tf.matmul(X, tf.transpose(W1)), b1)

    return Z1


def compute_cost(Z1, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z1, labels=Y))
    return cost


def cost_fig(costs, learning_rate):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return True


def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def data_check(data):
    res = list(np.argmax(data, 1))
    num = len(res)
    classes = data.shape[1]
    for i in range(classes):
        print(str(i) + '的比例', round(100.0 * res.count(i) / num, 2), '%')
    print('<------------------分割线---------------------->')


def model(X_train, Y_train, X_test, Y_test, file, epochs=2000, minibatch_size=64, learning_rate=0.5, print_cost=True):
    ops.reset_default_graph()
    m, n_x = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters(n_x, n_y, file)

    Z1 = forward_propagation(X, parameters)

    # cost = compute_cost(Z1, Y)
    cost = compute_cost(Z1, Y)+tf.contrib.layers.l1_regularizer(.2)(parameters['W1'])
    # tf.nn.l2_loss(parameters['W1'])
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.99).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                z1, par, _, temp_cost = sess.run([Z1, parameters, optimizer, cost],
                                                 feed_dict={X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches

            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, temp_cost))
            if print_cost and epoch % 10 == 0:
                costs.append(minibatch_cost)

        cost_fig(costs, learning_rate)

        predict_op = tf.argmax(Z1, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

    return par


if __name__ == '__main__':
    file = 'face_data_3_64'
    load_data(file)

    data_train = scio.loadmat(file + '_train')
    X_train = data_train['X']
    Y_train = data_train['Y']

    data_test = scio.loadmat(file + '_test')
    X_test = data_test['X']
    Y_test = data_test['Y']

    # data_check(Y_test)
    # data_check(Y_train)
    parameters = model(X_train, Y_train, X_test, Y_test, file, epochs=500, learning_rate=0.005)
    # W1 = parameters['W1']
    # b1 = parameters['b1']
    # scio.savemat(file + '_parameter', {'W1': W1, 'b1': b1})
