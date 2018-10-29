from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import scipy.io as scio
import numpy as np
import math
from practice_one.model.utils import *
import scipy.io as scio


# Network Parameters


#num_classes = 9  # MNIST total classes (0-9 digits)


# tf Graph input
def creat_placeholder(timesteps, num_input, num_classes):
    X = tf.placeholder(tf.float32, [None, timesteps, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    return X, Y


def preprocessing(Xtr, Xte, Ytr, Yte):
    Xtr = Xtr.reshape([-1, 14, 2])
    Xte = Xte.reshape([-1, 14, 2])
    return Xtr / 255., Xte / 255., Ytr, Yte


# Define weights
def creat_parameters(num_hidden, num_classes):
    weights = {
        # Hidden layer weights => 2*n_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    return weights, biases


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


def BiRNN(x, weights, biases, num_hidden=128, timesteps=14):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, timesteps, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    # Apply Dropout
    # outputs = tf.nn.dropout(x=outputs, keep_prob=0.95)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out'], outputs


def model(X_train, Y_train, X_test, Y_test, num_hidden=128, learning_rate=0.001, training_steps=300, display_step=100,
          batch_size=128):
    m, n_x0, n_x1 = X_train.shape
    n_y = Y_train.shape[1]
    X, Y = creat_placeholder(n_x0, n_x1, n_y)

    weights, biases = creat_parameters(num_hidden, n_y)
    parameter = {}
    parameter["weights"] = weights
    parameter["biases"] = biases
    l2_loss = tf.nn.l2_loss(weights['out'])

    logits ,outputs= BiRNN(X, weights, biases, num_hidden, n_x0)
    prediction = tf.nn.softmax(logits)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y)) + 0.002 * l2_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    accept_train_accuracy = 0.
    accept_test_accuracy = 0.

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for step in range(1, training_steps + 1):
            minibatch_cost = 0.
            num_minibatches = int(m / batch_size)
            minibatches = random_mini_batches(X_train, Y_train, batch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, par = sess.run([train_op, parameter], feed_dict={X: minibatch_X, Y: minibatch_Y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: minibatch_X,
                                                                         Y: minibatch_Y})
                    print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

        print("Optimization Finished!")
        scio.savemat("rnn_parameters.mat", par)
        pre_tr, train_logits, accu = sess.run([prediction, logits, accuracy], feed_dict={X: X_train, Y: Y_train})
        pre_te, test_logits,outputs, accu_test = sess.run([prediction, logits, outputs,accuracy], feed_dict={X: X_test, Y: Y_test})

        scio.savemat('train_logits.mat', {"res": train_logits})
        scio.savemat('test_logits.mat', {"res": test_logits})

        Y_tr = np.argmax(pre_tr, 1)
        Y_te = np.argmax(pre_te, 1)
        print(Y_te)
        print(outputs)

        for i in range(3):
            print(str(i) + '的比例', round(100.0 * list(Y_tr).count(i) / len(Y_tr), 2), '%')
            print(str(i) + "的比例", round(100. * list(Y_te).count(i) / len(Y_te), 2), "%")

        # print(len(np.argmax(Y_train)))
        for j in range(len(Y_tr)):
            if Y_tr[j] in accept_ans[np.argmax(Y_train[j])]:
                accept_train_accuracy += 1. / len(Y_tr)

        for k in range(len(Y_te)):
            if Y_te[k] in accept_ans[np.argmax(Y_test[k])]:
                accept_test_accuracy += 1. / len(Y_te)

        # Calculate accuracy for 128 mnist test images
        print("Train Accuracy:", accu)
        print("Train Accept Accuracy:", accept_train_accuracy)
        print("Testing Accuracy:", accu_test)
        print("Test Accept Accuracy:", accept_test_accuracy)


if __name__ == '__main__':
    name = 'Syh'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY64_expend'
    elif name == 'Syh':
        file = 'face_1_channel_sense'

    # load data
    X_train, X_test, Y_train, Y_test = load_data(file)

    # preprocess
    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)

    num_hidden = 128  # hidden layer num of features

    model(X_train, Y_train, X_test, Y_test, learning_rate=0.001)

    data_check(Y_train)
