# coding:utf-8
'''
Created on 2017/12/6

@author: sunyihuan
'''
from __future__ import print_function

import numpy as np
import tensorflow as tf

import scipy.io as scio
from sklearn.model_selection import train_test_split

accept_ans = [
    [0, 1, 3],
    [1, 0, 2, 4],
    [2, 1, 5],
    [3, 0, 4, 6],
    [4, 1, 3, 5, 7],
    [5, 2, 4, 8],
    [6, 3, 7],
    [7, 6, 4, 8],
    [8, 7, 5],
]


def main(Xtr, Ytr, Xte, Yte):
    xtr = tf.placeholder(tf.float32, [None, 16384])
    xte = tf.placeholder(tf.float32, [16384])
    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.arg_min(distance, 0)

    accuracy = 0.

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        # loop over test data
        for i in range(len(Xte)):
            # Get nearest neighbor
            nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
            # Get nearest neighbor class label and compare it to its true label
            print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), "True Class:", np.argmax(Yte[i]))
            # Calculate accuracy
            if np.argmax(Ytr[nn_index]) in accept_ans[np.argmax(Yte[i])]:
                accuracy += 1. / len(Xte)
        print("Done!")
        print("Accuracy:", accuracy)


if __name__ == '__main__':
    name = 'Syh'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY'
    elif name == 'Syh':
        file = 'E:/deeplearning_Data/face_1_channel_XY'

    data_train = scio.loadmat(file)

    X_train, X_test, Y_train, Y_test = train_test_split(data_train['X'], data_train['Y'], test_size=0.2)
    X_train = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])


    main(X_train, Y_train, X_test, Y_test)
