# coding:utf-8 
'''
created on 

@author:sunyihuan
'''

import scipy.io as scio
import tensorflow as tf
from tensorflow.contrib import rnn
from practice_one.model.utils import *
import numpy as np

parameters = scio.loadmat("rnn_parameters")

print(len(parameters["weights"]["out"][0][0]))
# print(parameters["biases"]["out"][0][0])


def creat_placeholder(timesteps, num_input, num_classes):
    X = tf.placeholder(tf.float32, [None, timesteps, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    return X, Y


def creat_parameters(num_hidden, num_classes):
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    return weights, biases


def m_rnn(X, num_hidden=128, timesteps=14):
    x = tf.transpose(X, [1, 0, 2])
    x = tf.reshape(x, [-1, timesteps])
    x = tf.split(x, 2)
    # x = tf.unstack(X, timesteps, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)


    # Get lstm cell output
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    # last = tf.gather(outputs, tf.shape(outputs)[0] - 1)

    return outputs


def model(X_train, X_test, Y_train, Y_test, num_hidden=128, timesteps=14):
    # m, n_x0, n_x1 = X_train.shape
    # n_y = Y_train.shape[1]
    # X, Y = creat_placeholder(n_x0, n_x1, n_y)
    X = tf.constant(X_test, dtype=tf.float32)
    # print("X", X)
    outputs = m_rnn(X, num_hidden=num_hidden, timesteps=timesteps)
    # print("ouputs", outputs)
    weights = tf.constant(parameters["weights"]["out"][0][0], dtype=tf.float32)
    biases = tf.constant(parameters["biases"]["out"][0][0], tf.float32)
    logists = tf.matmul(outputs[-1], weights) + biases
    # print("logists", logists)
    pre = tf.nn.softmax(logists)
    # print("pre", pre)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pre2 = pre.eval()
        b1 = sess.run(outputs)
        results = np.squeeze(b1)
        print("length of outputs", len(results))
        print(b1[1].shape)
        print("pre2",pre2)

    return True


if __name__ == "__main__":
    file = "face_1_channel_sense"
    X_train, X_test, Y_train, Y_test = load_data(file)
    X_train, X_test, Y_train, Y_test = X_train.reshape([-1, 14, 2]) / 255., X_test.reshape(
        [-1, 14, 2]) / 255., Y_train, Y_test
    print(X_test.shape)

    model(X_train, X_test, Y_train, Y_test)
