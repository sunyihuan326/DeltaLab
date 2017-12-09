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
from sklearn.model_selection import train_test_split


def init_sets(X, Y, file, distribute):
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    assert len(distribute) == 2
    assert sum(distribute) == 1
    scio.savemat(file + 'SoftMax_train',
                 {'X': shuffled_X[:, :int(m * distribute[0])], 'Y': shuffled_Y[:, :int(m * distribute[0])]})
    scio.savemat(file + 'SoftMax_test',
                 {'X': shuffled_X[:, int(m * distribute[0]):], 'Y': shuffled_Y[:, int(m * distribute[0]):]})
    return True


def load_data(file):
    if not os.path.exists(file + 'SoftMax_test.mat'):
        data = scio.loadmat(file)
        init_sets(data['X'].T, data['Y'].T, file, distribute=[0.8, 0.2])
    # X(784, 20000)
    # Y(10, 20000)
    return True


def initialize_parameters(n_x, n_y, file, ifExtend=False):
    W1 = tf.get_variable(name='W1', dtype=tf.float32, shape=(n_y, n_x),
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(dtype=tf.float32, name='b1', shape=(1),
                         initializer=tf.contrib.layers.xavier_initializer())
    # b1 = tf.constant(0.1)
    if ifExtend and os.path.exists(file + 'SoftMax_parameters'):
        parameters = scio.loadmat(file + 'SoftMax_parameters')
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
    # Z1 = tf.nn.dropout(Z1, 0.9)
    return Z1


def compute_cost(Z1, Y, parameters, regular=False):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z1, labels=Y))
    if regular:
        cost += tf.contrib.layers.l2_regularizer(.2)(parameters['W1'])
    return cost


def cost_fig(costs, learning_rate):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return True


def data_check(data):
    res = list(np.argmax(data.T, 1))
    num = len(res)
    classes = data.shape[0]
    for i in range(classes):
        print(str(i) + '的比例', round(100.0 * res.count(i) / num, 2), '%')
    print('<------------------分割线---------------------->')


def model(X_train, Y_train, X_test, Y_test, file, epochs=2000, learning_rate=0.5, print_cost=True):
    ops.reset_default_graph()
    n_x = X_train.shape[1]
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_x, n_y)
    print(X)
    print(Y)
    parameters = initialize_parameters(n_x, n_y, file)
    print('W1===================', parameters['W1'])
    print('b1==================', parameters['b1'])
    Z1 = forward_propagation(X, parameters)
    print('ZL========', Z1)

    cost = compute_cost(Z1, Y, parameters, False)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            z1, par, _, temp_cost = sess.run([Z1, parameters, optimizer, cost],
                                             feed_dict={X: X_train, Y: Y_train})
            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, temp_cost))
            if print_cost and epoch % 1 == 0:
                costs.append(temp_cost)

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
    name = 'Syh'
    if name == 'Dxq':
        file = 'F:/dataSets/MNIST/mnist_data_small'
    elif name == 'Syh':
        file = 'E:/deeplearning_Data/mnist_data_small'

    data_train = scio.loadmat(file)
    X_train, X_test, Y_train, Y_test = train_test_split(data_train['X'], data_train['Y'], test_size=0.2)

    # print(X_train.shape,X_test.shape,Y_train.shape)

    # data_check(Y_train)
    # data_check(Y_test)
    #
    parameters = model(X_train, Y_train, X_test, Y_test, file, epochs=20, learning_rate=0.01)
    W1 = parameters['W1']
    b1 = parameters['b1']
    scio.savemat(file + 'SoftMax_parameter', {'W1': W1, 'b1': b1})
