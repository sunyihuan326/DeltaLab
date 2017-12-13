# coding:utf-8
'''
Created on 2017/12/13

@author: sunyihuan
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest

# Ignore all GPUs, tf random forest does not benefit from it.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
from practice_one.model.utils import *
import numpy as np


def accept_accuracy(Y_tr, Y_train):
    '''
    
    :param Y_tr: 训练的标签
    :param Y_train: 实际标签
    :return accept_accuracy：可接受准确率
    '''
    accept_accuracy = 0.
    for j in range(len(Y_tr)):
        if Y_tr[j] in accept_ans[Y_train[j]]:
            accept_accuracy += 1. / len(Y_tr)
    return accept_accuracy


def model(X_train, X_test, Y_train, Y_test, num_features, max_nodes, num_steps, num_classes=9, num_trees=9,
          batch_size=128):
    # Input and Target data
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # For random forest, labels must be integers (the class id)
    Y = tf.placeholder(tf.int32, shape=[None])

    # Random Forest Parameters
    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                          num_features=num_features,
                                          num_trees=num_trees,
                                          max_nodes=max_nodes).fill()

    # Build the Random Forest
    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    # Get training graph and loss
    train_op = forest_graph.training_graph(X, Y)
    loss_op = forest_graph.training_loss(X, Y)

    # Measure the accuracy
    infer_op = forest_graph.inference_graph(X)
    correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init_vars = tf.global_variables_initializer()

    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars)

    # Training
    for i in range(1, num_steps + 1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, l = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: Y_train})
        if i % 20 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={X: X_train, Y: Y_train})
            print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

    # Test Model
    # test_x, test_y = mnist.test.images, mnist.test.labels
    pre_tr, accu = sess.run([infer_op, accuracy_op], feed_dict={X: X_train, Y: Y_train})
    pre_te, accu_test = sess.run([infer_op, accuracy_op], feed_dict={X: X_test, Y: Y_test})
    Y_tr = np.argmax(pre_tr, 1)
    Y_te = np.argmax(pre_te, 1)
    accept_train_accuracy = accept_accuracy(Y_tr, Y_train)
    accept_test_accuracy = accept_accuracy(Y_te, Y_test)
    print("Train Accuracy:%f,Train Accept Accuracy:%f" % (accu, accept_train_accuracy))
    print("Test Accuracy:%f,Test Accept Accuracy:%f" % (accu_test, accept_test_accuracy))


if __name__ == '__main__':
    # Parameters
    num_steps = 160  # Total steps to train
    batch_size = 128  # The number of samples per batch
    num_classes = 9  # The 10 digits
    num_features = 142  # Each image is 28x28 pixels
    num_trees = 9
    max_nodes = 700

    # mnist = input_data.read_data_sets("../tmp/data/", one_hot=False)
    X_train, X_test, Y_train, Y_test = load_data('E:/deeplearning_Data/face_1_channel_XY_points_expend')

    Y_train = np.argmax(Y_train, 1)
    Y_test = np.argmax(Y_test, 1)
    model(X_train, X_test, Y_train, Y_test, num_features=num_features, max_nodes=max_nodes, num_steps=num_steps)
