# coding:utf-8
'''
Created on 2017/11/15.

@author: chk01
'''
import os
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import math


def init_sets(X, Y, file, distribute):
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    assert len(distribute) == 2
    assert sum(distribute) == 1
    scio.savemat(file + 'DNN_train',
                 {'X': shuffled_X[:, :int(m * distribute[0])], 'Y': shuffled_Y[:, :int(m * distribute[0])]})
    scio.savemat(file + 'DNN_test',
                 {'X': shuffled_X[:, int(m * distribute[0]):], 'Y': shuffled_Y[:, int(m * distribute[0]):]})
    return True


def load_data(file):
    if not os.path.exists(file + 'DNN_test.mat'):
        data = scio.loadmat(file)
        m = data['X'].shape[0]
        x = data['X'].reshape(-1, m)
        y = np.squeeze(data['Y']).T
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
        print('here')
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


def model(X_train, Y_train, X_test, Y_test, layer_dims, epochs=2000, minibatch_size=64, initial_learning_rate=0.5,
          print_cost=True):
    ops.reset_default_graph()
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters_deep(layer_dims)
    global_step = tf.Variable(0, trainable=False)
    # layer_dims = [5, 4, 3]
    ZL = forward_propagation(X, parameters)

    correct_pred = tf.equal(tf.argmax(tf.transpose(ZL), 1), tf.argmax(tf.transpose(Y), 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    cost = compute_cost(ZL, Y)
    tf.summary.scalar('cost', cost)
    # cost = compute_cost(Z1, Y) + tf.contrib.layers.l1_regularizer(.2)(parameters['W1'])
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.99).minimize(cost)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=100, decay_rate=0.98)
    tf.summary.scalar('learning_rate', learning_rate+0.001)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    add_global = global_step.assign_add(1)
    merge_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(logdir='logdir/MNIST/DNN')
        for epoch in range(epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                summary, zl, par, _, temp_cost, _, acc = sess.run(
                    [merge_op, ZL, parameters, optimizer, cost, add_global, accuracy],
                    feed_dict={X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
                writer.add_summary(summary)
            if print_cost and epoch % 5 == 0:
                print("Cost|Acc after epoch %i: %f | %f" % (epoch, temp_cost, acc))

        predict_op = tf.argmax(tf.transpose(ZL), 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(tf.transpose(Y), 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

    return par


if __name__ == '__main__':
    name = 'Dxq'
    if name == 'Dxq':
        file = 'F:/dataSets/MNIST/mnist_data_small'
    elif name == 'Syh':
        file = ''
    load_data(file)

    data_train = scio.loadmat(file + 'DNN_train')
    X_train = data_train['X']
    Y_train = data_train['Y']
    # print(X_train.shape)
    # (784, 16000)
    # print(Y_train.shape)
    data_test = scio.loadmat(file + 'DNN_test')
    X_test = data_test['X']
    Y_test = data_test['Y']

    layer_dims = [784, 10]
    data_check(Y_test)
    data_check(Y_train)
    parameters = model(X_train, Y_train, X_test, Y_test, layer_dims, epochs=200, initial_learning_rate=0.5)
    W1 = parameters['W1']
    b1 = parameters['b1']
    scio.savemat(file + 'DNN_parameter', {'W1': W1, 'b1': b1})
