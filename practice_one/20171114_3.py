# coding:utf-8
'''
Created on 2017/11/14.

@author: chk01
'''
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as scio
import os


def init_sets(X, Y, file, distribute):
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]
    assert len(distribute) == 2
    assert sum(distribute) == 1
    scio.savemat(file + 'train',
                 {'X': shuffled_X[:int(m * distribute[0]), :, :, :], 'Y': shuffled_Y[:int(m * distribute[0]), :]})
    scio.savemat(file + 'test',
                 {'X': shuffled_X[int(m * distribute[0]):, :, :, :], 'Y': shuffled_Y[int(m * distribute[0]):, :]})
    return True


def random_mini_batches(X, Y, mini_batch_size=64,seed=0):
    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, int(num_complete_minibatches) + 1):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # if m % mini_batch_size != 0:
    #     mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
    #     mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
    #     mini_batch = (mini_batch_X, mini_batch_Y)
    #     mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(name='X', shape=(None, n_H0, n_W0, n_C0), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(None, n_y), dtype=tf.float32)

    return X, Y


def initialize_parameters(ifExtend=False):
    W1 = tf.get_variable(name='W1', dtype=tf.float32, shape=(4, 4, 3, 8),
                         initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable(name='W2', dtype=tf.float32, shape=(2, 2, 8, 16),
                         initializer=tf.contrib.layers.xavier_initializer())

    # if ifExtend and os.path.exists('face_data_3_64_parameters'):
    #     parameters = scio.loadmat('face_data_3_64_parameters')
    #     W1 = tf.Variable(parameters['W1'])
    #     W2 = tf.Variable(parameters['W2'])

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']

    Z1 = tf.nn.conv2d(input=X, filter=W1, strides=(1, 1, 1, 1), padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(value=A1, ksize=(1, 4, 4, 1), strides=(1, 4, 4, 1), padding='SAME')
    # P1 = tf.nn.dropout(P1, 0.9)

    Z2 = tf.nn.conv2d(input=P1, filter=W2, strides=(1, 1, 1, 1), padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(value=A2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
    # P2 = tf.nn.dropout(P2, 0.9)
    PP2 = tf.contrib.layers.flatten(inputs=P2)
    Z3 = tf.contrib.layers.fully_connected(PP2, 9, activation_fn=None)

    return P1, P2, Z3


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.001,
          num_epochs=100, minibatch_size=64, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost
    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    P1, P2, Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    # + tf.contrib.layers.l1_regularizer(0.99)(parameters['W1'])
    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size,seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                z3, p1, p2, par, _, temp_cost = sess.run([Z3, P1, P2, parameters, optimizer, cost],
                                                         feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###

                minibatch_cost += temp_cost / num_minibatches
            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title(
            "Learning rate =" + str(learning_rate) + "num_epochs" + str(num_epochs))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy, par


def init_data(file, distribution):
    data = scio.loadmat(file)
    x = data['X'].reshape(-1, 64, 64, 3)
    y = np.squeeze(data['Y']).T
    init_sets(x, y, file, distribution)


def data_check(data):
    res = list(np.argmax(data, 1))
    num = len(res)
    classes = data.shape[1]
    for i in range(classes):
        print(str(i) + '的比例', round(100.0 * res.count(i) / num, 2), '%')


if __name__ == '__main__':
    tf.reset_default_graph()
    # (None, 64, 64, 3)
    # init_data('face_data_3_64', [0.8, 0.2])
    data_train = scio.loadmat('face_data_3_64train')
    X_train = data_train['X']
    Y_train = data_train['Y']

    data_test = scio.loadmat('face_data_3_64test')
    X_test = data_test['X']
    Y_test = data_test['Y']

    # data_check(Y_test)
    # data_check(Y_train)

    _, _, parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=100)
    W1 = parameters['W1']
    W2 = parameters['W2']
    scio.savemat('face_data_3_64_parameters.mat', {'W1': W1, 'W2': W2})
