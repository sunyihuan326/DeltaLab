# coding:utf-8
'''
Created on 2017/12/5.

@author: chk01
'''

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import scipy.io as scio
import os
import numpy as np
from tensorflow.python.framework import ops
import math
import matplotlib.pyplot as plt
from PIL import Image


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv_net(x, weights, biases, dropout):
    # x.shape (?, 16384)
    x = tf.reshape(x, shape=[-1, 128, 128, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    # print(conv1.shape==(?, 64, 64, 32))
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    # print(conv2.shape==(?, 32, 32, 64))
    # conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # conv3 = maxpool2d(conv3, k=1)
    # print(conv3.shape)
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    # print(fc1.shape == (?, 65536))
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


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


def load_data(file):
    if not os.path.exists(file + '_test.mat'):
        data = scio.loadmat(file)
        data_check(data['Y'].T)
        m = data['X'].shape[0]
        x = data['X'].reshape(m, -1).T
        y = np.squeeze(data['Y']).T
        init_sets(x, y, file, distribute=[0.8, 0.2])
    return True


def data_check(data):
    res = list(np.argmax(data.T, 1))
    num = len(res)
    classes = data.shape[0]
    for i in range(classes):
        print(str(i) + '的比例', round(100.0 * res.count(i) / num, 2), '%')
    print('<------------------分割线---------------------->')


def create_placeholders(n_x, n_y):
    X = tf.placeholder(name='X', shape=(None, n_x), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(None, n_y), dtype=tf.float32)

    return X, Y


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


def initialize_parameters(n_y):
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([32 * 32 * 64, 1024])),
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_y]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([64])),
        'bc3': tf.Variable(tf.random_normal([128])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_y]))
    }
    return weights, biases


def cost_fig(costs, learning_rate):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return True


def model(X_train, Y_train, X_test, Y_test, kp=1.0, epochs=2000, minibatch_size=64, initial_learning_rate=0.5,
          print_cost=True):
    ops.reset_default_graph()
    n_x, m = X_train.shape
    # 16384,1201
    n_y = Y_train.shape[0]
    # 9
    costs = []
    weights, biases = initialize_parameters(n_y)
    global_step = tf.Variable(0, trainable=False)
    X, Y = create_placeholders(n_x, n_y)
    # (?, 16384) (?, 9)
    keep_prob = tf.placeholder(tf.float32)

    logits = conv_net(X, weights, biases, keep_prob)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))

    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10, decay_rate=0.9)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate + 0.01)
    train_op = optimizer.minimize(loss_op)
    add_global = global_step.assign_add(1)
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):

            minibatch_cost = 0.
            num_minibatches = m // minibatch_size
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                _, loss, acc, par, _ = sess.run([train_op, loss_op, accuracy, weights, add_global],
                                                feed_dict={X: minibatch_X.T,
                                                           Y: minibatch_Y.T,
                                                           keep_prob: kp})
                minibatch_cost += loss / num_minibatches

            if print_cost and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, loss))
            if print_cost and epoch % 1 == 0:
                costs.append(minibatch_cost)
            print("Step " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.4f}".format(minibatch_cost) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

        print("Optimization Finished!")

        cost_fig(costs, learning_rate)
        train_accuracy = accuracy.eval({X: X_train.T, Y: Y_train.T, keep_prob: 1})
        test_accuracy = accuracy.eval({X: X_test.T, Y: Y_test.T, keep_prob: 1})
        print('train_accuracy', train_accuracy)
        print('test_accuracy', test_accuracy)
    return par


def draw_pic(data):
    data = data.reshape([128, 128])
    Image.fromarray(data).show()


if __name__ == '__main__':
    file = 'F:/dataSets/FaceChannel1/face_1_channel_XY'
    load_data(file)

    data_train = scio.loadmat(file + '_train')
    X_train = data_train['X']
    Y_train = data_train['Y']

    data_test = scio.loadmat(file + '_test')
    X_test = data_test['X']
    Y_test = data_test['Y']

    # data_check(Y_test)
    # data_check(Y_train)

    parameters = model(X_train, Y_train, X_test, Y_test, kp=1, epochs=10, initial_learning_rate=0.5)