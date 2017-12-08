# coding:utf-8
'''
Created on 2017/12/8

@author: sunyihuan
'''

import pandas as pd
import tensorflow as tf
import numpy as np
import scipy.io as sio

cluster_out = pd.read_csv('cluster_outlin.csv')
cluster_sen = pd.read_csv('cluster_sense.csv')

cluster_labels = []
cluster_out = cluster_out.values
cluster_sen = cluster_sen.values

for i in range(cluster_out.shape[0]):
    c = 3 * cluster_out[i][1] + cluster_sen[i][1]
    cluster_labels.append(c)

Y_data = sio.loadmat('E:/deeplearning_Data/face_1_channel_XY')
Y_train = Y_data['Y'][1200:, :]

Y = tf.placeholder(tf.float32, shape=[None, 9])
correct_prediction = tf.equal(cluster_labels, tf.cast(tf.argmax(Y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    correct_prediction, accuracy_op = sess.run([correct_prediction, accuracy_op], feed_dict={Y: Y_train})
    print("Test Accuracy:", accuracy_op)
