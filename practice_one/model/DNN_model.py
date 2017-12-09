# coding:utf-8
'''
Created on 2017/11/15.

@author: chk01
'''
import tensorflow as tf
from tensorflow.python.framework import ops

from practice_one.model.utils import *


def preprocessing(trX, teX, trY, teY):
    return trX, teX, trY, teY


def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable(dtype=tf.float32, name='W' + str(l),
                                                   shape=(layer_dims[l], layer_dims[l - 1]),
                                                   initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable(dtype=tf.float32, name='b' + str(l),
                                                   shape=(1, layer_dims[l]),
                                                   initializer=tf.contrib.layers.xavier_initializer())
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (1, layer_dims[l]))

    return parameters


def create_placeholders(n_x, n_y):
    X = tf.placeholder(name='X', shape=(None, n_x), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(None, n_y), dtype=tf.float32)

    return X, Y


def forward_propagation(X, parameters, kp):
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A = tf.nn.relu(tf.add(tf.matmul(A_prev, tf.transpose(W)), b))
        A = tf.layers.batch_normalization(A, axis=-1)
        A = tf.nn.dropout(A, kp)
    ZL = tf.add(tf.matmul(A, tf.transpose(parameters['W' + str(L)])), parameters['b' + str(L)])
    return ZL


def model(X_train, Y_train, X_test, Y_test, layer_dims, keep_prob=1.0, epochs=2000, minibatch_size=64,
          initial_learning_rate=0.5, minest_learning_rate=0.01):
    ops.reset_default_graph()

    m, n_x = X_train.shape
    n_y = Y_train.shape[1]

    kp = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters_deep(layer_dims)

    ZL = forward_propagation(X, parameters, kp)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ZL, labels=Y))
    tf.summary.scalar('cost', cost)
    # cost = compute_cost(Z1, Y) + tf.contrib.layers.l1_regularizer(.2)(parameters['W1'])
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.99).minimize(cost)

    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10, decay_rate=0.9)
    learning_rate = tf.maximum(learning_rate, minest_learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    add_global = global_step.assign_add(1)
    merge_op = tf.summary.merge_all()

    correct_pred = tf.equal(tf.argmax(ZL, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

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
                    feed_dict={X: minibatch_X, Y: minibatch_Y, kp: keep_prob})
                minibatch_cost += temp_cost / num_minibatches
                writer.add_summary(summary)
            if epoch % 5 == 0:
                print("Cost|Acc after epoch %i: %f | %f" % (epoch, temp_cost, acc))

        predict_op = tf.argmax(ZL, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train, kp: 1})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test, kp: 1})

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

    return par


if __name__ == '__main__':
    name = 'Dxq'
    if name == 'Dxq':
        file = 'F:/dataSets/MNIST/mnist_data_small'
    elif name == 'Syh':
        file = ''
    # load data
    X_train, X_test, Y_train, Y_test = load_data(file, test_size=0.2)
    # preprocessing
    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)

    data_check(Y_train)
    data_check(Y_test)

    layer_dims = [784, 64, 10]

    parameters = model(X_train, Y_train, X_test, Y_test, layer_dims, keep_prob=1.0, epochs=20,
                       initial_learning_rate=0.5)
    scio.savemat(file + 'DNN_parameter', parameters)
