# coding:utf-8 
'''
created on 2018/3/3

@author:Dxq
'''
# coding:utf-8
'''
Created on 2017/11/16.

@author: chk01
'''
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import scipy.io as scio
import math


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def init_sets(X, Y, file, distribute):
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    assert len(distribute) == 2
    assert sum(distribute) == 1
    scio.savemat(file + '_train',
                 {'X': shuffled_X[:, :int(m * distribute[0])], 'Y': shuffled_Y[:, :int(m * distribute[0])]})
    scio.savemat(file + '_test',
                 {'X': shuffled_X[:, int(m * distribute[0]):], 'Y': shuffled_Y[:, int(m * distribute[0]):]})
    return True


def load_data(x, y, file):
    init_sets(x, y, file, distribute=[0.8, 0.2])
    return True


def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable(dtype=tf.float32, name='W' + str(l),
                                                   shape=(layer_dims[l], layer_dims[l - 1]),
                                                   initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable(dtype=tf.float32, name='b' + str(l),
                                                   shape=(layer_dims[l], 1),
                                                   initializer=tf.contrib.layers.xavier_initializer())
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def create_placeholders(n_x, n_y):
    X = tf.placeholder(name='X', shape=(n_x, None), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(n_y, None), dtype=tf.float32)

    return X, Y


def forward_propagation(X, parameters):
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A = tf.nn.relu(tf.add(tf.matmul(W, A_prev), b))
        # A = tf.nn.dropout(A, 0.9)
        # 94
    ZL = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])

    return ZL


def compute_cost(ZL, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(ZL), labels=tf.transpose(Y)))
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
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def data_check(data):
    res = list(np.argmax(data.T, 1))
    num = len(res)
    classes = data.shape[0]
    for i in range(classes):
        print(str(i) + '的比例', round(100.0 * res.count(i) / num, 2), '%')
    print('<------------------分割线---------------------->')


def model(X_train, Y_train, X_test, Y_test, layer_dims, epochs=2000, minibatch_size=64, learning_rate=0.5,
          print_cost=True):
    ops.reset_default_graph()
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters_deep(layer_dims)
    # layer_dims = [5, 4, 3]
    ZL = forward_propagation(X, parameters)

    cost = compute_cost(ZL, Y)
    # cost = compute_cost(Z1, Y) + tf.contrib.layers.l1_regularizer(.2)(parameters['W1'])
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
                zl, par, _, temp_cost = sess.run([ZL, parameters, optimizer, cost],
                                                 feed_dict={X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches

            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, temp_cost))
            if print_cost and epoch % 1 == 0:
                costs.append(minibatch_cost)

        cost_fig(costs, learning_rate)

        predict_op = tf.argmax(tf.transpose(ZL), 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(tf.transpose(Y), 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

    return par


def predict(X, parameters):
    L = len(parameters) // 2
    params = {}
    for l in range(1, L):
        params['W' + str(l)] = tf.convert_to_tensor(parameters['W' + str(l)])
        params['b' + str(l)] = tf.convert_to_tensor(parameters['b' + str(l)])

    x = tf.placeholder("float", [784, None])

    z3 = forward_propagation(x, params)
    p = tf.argmax(tf.transpose(z3), 1)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})
    result = pd.read_csv('C:/Users/chk01/Desktop/sample_submission.csv')
    result['Label'] = prediction
    result.to_csv('result.csv')
    return prediction


if __name__ == '__main__':
    file = 'kaggle_mnist'
    # parameters = scio.loadmat(file + '_parameter')
    # result = pd.read_csv("C:/Users/chk01/Desktop/test.csv")
    # result_X = result.values.T[0:]
    # print(result_X.shape)
    # predict(result_X, parameters)
    # assert 1 == 0

    train = pd.read_csv("C:/Users/chk01/Desktop/train.csv")
    train_X = train.values.T[1:]
    train_Y = train['label'].values.reshape(1, -1)
    train_Y = convert_to_one_hot(train_Y, 10)

    load_data(train_X, train_Y, 'kaggle_mnist')

    data_train = scio.loadmat(file + '_train')
    X_train = data_train['X']
    Y_train = data_train['Y']
    # print(X_train.shape)
    # (784, 16000)
    # print(Y_train.shape)
    data_test = scio.loadmat(file + '_test')
    X_test = data_test['X']
    Y_test = data_test['Y']

    layer_dims = [784, 10]
    data_check(Y_test)
    data_check(Y_train)
    parameters = model(X_train, Y_train, X_test, Y_test, layer_dims, epochs=200, learning_rate=0.001)

    scio.savemat(file + '_parameter', parameters)
