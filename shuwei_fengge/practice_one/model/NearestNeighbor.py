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
from practice_one.model.utils import *
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report


def preprocessing(trX, teX, trY, teY):
    res = RandomOverSampler(random_state=42)
    trY = np.argmax(trY, 1)
    teY = np.argmax(teY, 1)
    trX, trY = res.fit_sample(trX, trY)
    teX, teY = res.fit_sample(teX, teY)

    trY = np.eye(3)[trY]
    teY = np.eye(3)[teY]
    return trX, teX, trY, teY


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
    m, n_x = Xtr.shape
    xtr = tf.placeholder(tf.float32, [None, n_x])
    xte = tf.placeholder(tf.float32, [n_x])
    distance = tf.reduce_sum(tf.square(x=tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.argmin(distance, 0)
    # pred3 = tf.nn.top_k(-distance, k=1)

    accept_accuracy = 0.
    accuracy = 0.

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        classes = []

        # loop over test data
        for i in range(len(Xte)):
            # Get nearest neighbor
            nn3_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
            # Get nearest neighbor class label and compare it to its true label
            if i % 10 == 0:
                print("Test", i, "Prediction:", np.argmax(Ytr[nn3_index]), "True Class:", np.argmax(Yte[i]))
            # pre3 = nn3_index.indices

            # Calculate accuracy
            if np.argmax(Ytr[nn3_index] == np.argmax(Yte[i])):
                accuracy += 1. / len(Xte)
            if abs(np.argmax(Ytr[nn3_index]) - np.argmax(Yte[i])) > 1:
                accept_accuracy += 1. / len(Xte)
            classes.append(np.argmax(Ytr[nn3_index]))

        print("Done!")
        # print("Accept Accuracy:", accept_accuracy)
        print("Accuracy:", accuracy)
        print("Accept accuracy error:", accept_accuracy)
        return classes


if __name__ == '__main__':
    name = 'Syh'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY'
    elif name == 'Syh':
        file = 'face_1_channel_sense'

    X_train, X_test, Y_train, Y_test = load_data(file)
    # X_train = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
    # X_test = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])

    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)

    for i in range(3):
        print(str(i) + '的比例', round(100.0 * list(np.argmax(Y_train, 1)).count(i) / len(np.argmax(Y_train, 1)), 2), '%')

    classes = main(X_train, Y_train, X_test, Y_test)
    for i in range(3):
        print(str(i) + '的比例', round(100.0 * list(classes).count(i) / len(classes), 2), '%')


    print(classification_report(y_true=np.argmax(Y_test, 1), y_pred=classes, target_names=["0", "1", "2"]))
