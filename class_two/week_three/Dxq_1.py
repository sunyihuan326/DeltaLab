# coding:utf-8
'''
Created on 2017/11/7.

@author: chk01
'''
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from class_two.week_three.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)


def exam1():
    y_hat = tf.constant(36, name='Y-hat')
    y = tf.constant(39, name='y')
    loss = tf.Variable((y - y_hat) ** 2, name='loss')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(loss))


def exam2():
    a = tf.constant(2)
    b = tf.constant(3)
    c = tf.multiply(a, b)
    return c


def exam3(x_input):
    with tf.Session() as sess:
        x = tf.placeholder(tf.int64, name='x')
        y = 2 * x
        print(sess.run(y, feed_dict={x: x_input}))


# GRADED FUNCTION: linear_function

def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)

    X = tf.constant(np.random.randn(3, 1), tf.float32, name='X')
    W = tf.constant(np.random.randn(4, 3), tf.float32, name='W')
    b = tf.constant(np.random.randn(4, 1), tf.float32, name='b')
    Y = tf.matmul(W, X) + b

    with tf.Session() as sess:
        result = sess.run(Y)

    return result


# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    results -- the sigmoid of z
    """

    x = tf.placeholder(tf.float32, name='x')

    sigmoid = tf.nn.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: z})

    return result


def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    
    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)

    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation. So logits will feed into z, and labels into y.
    
    Returns:
    cost -- runs the session of the cost (formula (2))
    """

    z = tf.placeholder(tf.float32, name='z-input')
    y = tf.placeholder(tf.float32, name='y-input')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    with tf.Session() as sess:
        cost = sess.run(cost, feed_dict={z: logits, y: labels})

    return cost


# GRADED FUNCTION: one_hot_matrix

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """

    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    tf.nn.sigmoid_cross_entropy_with_logits()
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)

    return one_hot


if __name__ == '__main__':
    # exam1()
    logits = np.array([0.2, 0.4, 0.7, 0.9])
    cost = cost(logits, np.array([0, 0, 1, 1]))
    print("cost = " + str(cost))
    tf.one_hot(labels,C,axis=0)