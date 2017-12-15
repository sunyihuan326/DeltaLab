# coding:utf-8
'''
Created on 2017/12/5.

@author: chk01
'''

from __future__ import division, print_function, absolute_import

import tensorflow as tf
from tensorflow.python.framework import ops

from practice_one.model.utils import *


def preprocessing(trX, teX, trY, teY):
    return trX / 255., teX / 255., trY, teY


def create_placeholders(n_x, n_y):
    X = tf.placeholder(name='X', shape=(None, n_x), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(None, n_y), dtype=tf.float32)

    return X, Y


def initialize_parameters(n_y):
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # 5x5 conv, 32 inputs, 64 outputs
        'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([16 * 16 * 64, 1024])),
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
    x = tf.reshape(x, shape=[-1, 64, 64, 1])
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


def model(X_train, Y_train, X_test, Y_test, keep_prob=1.0, epochs=2000, minibatch_size=64,
          initial_learning_rate=0.5, minest_learning_rate=0.01):
    ops.reset_default_graph()

    m, n_x = X_train.shape
    n_y = Y_train.shape[1]

    kp = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    X, Y = create_placeholders(n_x, n_y)
    weights, biases = initialize_parameters(n_y)

    logits = conv_net(X, weights, biases, kp)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))

    tf.summary.scalar('loss', loss_op)

    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10, decay_rate=0.9)
    learning_rate = tf.maximum(learning_rate, minest_learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    add_global = global_step.assign_add(1)

    # Evaluate model
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    summary_merge_op = tf.summary.merge_all()

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        summary_write = tf.summary.FileWriter('logdir/CNN_model', sess.graph)
        for epoch in range(epochs):
            minibatch_cost = 0.
            num_minibatches = m // minibatch_size
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                minibatch_X, minibatch_Y = minibatch
                summary_merge, _, loss, acc, par, _ = sess.run(
                    [summary_merge_op, train_op, loss_op, accuracy, weights, add_global],
                    feed_dict={X: minibatch_X, Y: minibatch_Y, kp: keep_prob})
                minibatch_cost += loss / num_minibatches

            if epoch % 5 == 0:
                summary_write.add_summary(summary_merge, epoch)
                print("Cost after epoch %i: %f" % (epoch, loss))

            print("Step " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.4f}".format(minibatch_cost) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

        print("Optimization Finished!")

        train_accuracy = accuracy.eval({X: X_train.T, Y: Y_train.T, kp: 1})
        test_accuracy = accuracy.eval({X: X_test.T, Y: Y_test.T, kp: 1})
        print('train_accuracy', train_accuracy)
        print('test_accuracy', test_accuracy)
    return par


if __name__ == '__main__':
    name = 'Dxq'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY64'
    elif name == 'Syh':
        file = 'E:/deeplearning_Data/face_1_channel_XY64'

    # load data
    X_train, X_test, Y_train, Y_test = load_data(file)

    # preprocess
    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)
    print(X_train.shape, Y_train.shape)

    # check the distribution
    # data_check(Y_train)
    # data_check(Y_test)

    parameters = model(X_train, Y_train, X_test, Y_test, keep_prob=1, epochs=100, initial_learning_rate=0.5)
    scio.savemat(file + '64CNN_parameter', parameters)
