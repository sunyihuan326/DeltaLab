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
    scio.savemat(file + 'DNN2_train',
                 {'X': shuffled_X[:, :int(m * distribute[0])], 'Y': shuffled_Y[:, :int(m * distribute[0])]})
    scio.savemat(file + 'DNN2_test',
                 {'X': shuffled_X[:, int(m * distribute[0]):], 'Y': shuffled_Y[:, int(m * distribute[0]):]})
    return True


def load_data(file):
    if not os.path.exists(file + 'DNN2_test.mat'):
        data = scio.loadmat(file)
        m = data['X'].shape[0]
        x = data['X'].reshape(m, -1).T
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


def forward_propagation(X, parameters, keep_prob):
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A_prev = tf.layers.batch_normalization(A_prev, axis=0)
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A = tf.nn.relu(tf.add(tf.matmul(W, A_prev), b))
        A = tf.nn.dropout(A, keep_prob)
    ZL = tf.add(tf.matmul(parameters['W' + str(L)], A), parameters['b' + str(L)])
    ZL = tf.layers.batch_normalization(ZL, axis=0)
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


def accuracy_cal(train_pre_val, train_cor_val):
    accept_ans = [
        [0, 1, 3],
        [1, 0, 2, 4],
        [2, 1, 5],
        [3, 0, 4, 6],
        [4, 1, 3, 5, 7],
        [5, 2, 4, 8],
        [6, 3, 7],
        [7, 6, 4, 8],
        [8, 7, 5],
    ]
    error_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, len(train_cor_val), 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct = 0
    real_correct = 0
    pre_real = np.zeros([len(train_cor_val), 2])
    for i in range(len(train_cor_val)):
        pre_real[i, :] = [train_pre_val[i], train_cor_val[i]]
        error_count[train_cor_val[i] + 10] += 1
        if train_pre_val[i] in accept_ans[train_cor_val[i]]:
            correct += 1
        else:
            error_count[train_cor_val[i]] += 1
        if train_pre_val[i] == train_cor_val[i]:
            real_correct += 1
    scio.savemat('result_analysis/DNN2/res', {'result': pre_real})
    scio.savemat('result_analysis/DNN2/error', {'result': error_count})
    return correct / len(train_cor_val), real_correct / len(train_cor_val)


def model(X_train, Y_train, X_test, Y_test, layer_dims, kp=1.0, epochs=2000, minibatch_size=64,
          initial_learning_rate=0.5, print_cost=True):
    ops.reset_default_graph()
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]
    # costs = []

    X, Y = create_placeholders(n_x, n_y)
    keep_prob = tf.placeholder(tf.float32)

    parameters = initialize_parameters_deep(layer_dims)
    global_step = tf.Variable(0, trainable=False)
    # layer_dims = [5, 4, 3]
    ZL = forward_propagation(X, parameters, keep_prob)

    cost = compute_cost(ZL, Y)
    tf.summary.scalar(name='cost', tensor=cost)
    predict_op = tf.argmax(tf.transpose(ZL), 1)
    correct_op = tf.argmax(tf.transpose(Y), 1)

    tf.summary.histogram(name='predict_op', values=predict_op)
    tf.summary.histogram(name='correct_op', values=correct_op)

    # cost = compute_cost(Z1, Y) + tf.contrib.layers.l1_regularizer(.2)(parameters['W1'])
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.99).minimize(cost)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10, decay_rate=0.95)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate + 0.001).minimize(cost)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    add_global = global_step.assign_add(1)
    init = tf.global_variables_initializer()

    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)

        summary = tf.summary.FileWriter(logdir='logdir/DNN2', graph=sess.graph)
        for epoch in range(epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                summary_op, zl, par, _, temp_cost, _ = sess.run(
                    [merged_summary_op, ZL, parameters, optimizer, cost, add_global],
                    feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: kp})
                summary.add_summary(summary_op, epoch)
                minibatch_cost += temp_cost / num_minibatches

            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, temp_cost))
                # if print_cost and epoch % 1 == 0:
                #     costs.append(minibatch_cost)

        # cost_fig(costs, learning_rate)

        train_pre_val = predict_op.eval({X: X_train, Y: Y_train, keep_prob: 1})
        train_cor_val = correct_op.eval({X: X_train, Y: Y_train, keep_prob: 1})
        train_accuracy, train_real_accuracy = accuracy_cal(train_pre_val, train_cor_val)

        test_pre_val = predict_op.eval({X: X_test, Y: Y_test, keep_prob: 1})
        test_cor_val = correct_op.eval({X: X_test, Y: Y_test, keep_prob: 1})
        test_accuracy, test_real_accuracy = accuracy_cal(test_pre_val, test_cor_val)

        print("Train Accuracy:", train_accuracy, train_real_accuracy)
        print("Test Accuracy:", test_accuracy, test_real_accuracy)

    return par


if __name__ == '__main__':
    name = 'Dxq'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY'
    elif name == 'Syh':
        file = 'E:/deeplearning_Data/face_1_channel_XY'
    load_data(file)

    data_train = scio.loadmat(file + 'DNN2_train')
    X_train = data_train['X'] / 255.
    Y_train = data_train['Y']

    data_test = scio.loadmat(file + 'DNN2_test')
    X_test = data_test['X'] / 255.
    Y_test = data_test['Y']
    layer_dims = [X_train.shape[0], 1024, Y_train.shape[0]]
    data_check(Y_test)
    data_check(Y_train)

    parameters = model(X_train, Y_train, X_test, Y_test, layer_dims, kp=1.0, epochs=200, initial_learning_rate=0.5)

    scio.savemat(file + 'DNN2_parameter', parameters)
