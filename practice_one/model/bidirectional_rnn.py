from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import scipy.io as scio
import numpy as np
import math
from practice_one.model.utils import *

'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''


# Network Parameters



# num_classes = 9  # MNIST total classes (0-9 digits)


# tf Graph input
def creat_placeholder(timesteps, num_input, num_classes):
    X = tf.placeholder(tf.float32, [None, timesteps, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    return X, Y


def preprocessing(Xtr, Xte, Ytr, Yte):
    Xtr = Xtr.reshape([-1, 64, 64])
    Xte = Xte.reshape([-1, 64, 64])
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


def BiRNN(x, weights, biases, num_hidden=128, timesteps=64):
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

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def model(X_train, Y_train, X_test, Y_test, num_hidden=128, learning_rate=0.001, training_steps=20, display_step=10,
          batch_size=64):
    m, n_x0, n_x1 = X_train.shape
    n_y = Y_train.shape[1]
    X, Y = creat_placeholder(n_x0, n_x1, n_y)

    weights, biases = creat_parameters(num_hidden, n_y)

    logits = BiRNN(X, weights, biases, num_hidden, n_x0)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

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
                sess.run(train_op, feed_dict={X: minibatch_X, Y: minibatch_Y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: minibatch_X,
                                                                         Y: minibatch_Y})
                    print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images

        print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: X_test, Y: Y_test}))


if __name__ == '__main__':
    name = 'Syh'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY64'
    elif name == 'Syh':
        file = 'E:/deeplearning_Data/face_1_channel_XY64'

    # load data
    X_train, X_test, Y_train, Y_test = load_data(file)
    # preprocess
    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)

    num_hidden = 110  # hidden layer num of features

    model(X_train, Y_train, X_test, Y_test, num_hidden=num_hidden, learning_rate=0.001)
