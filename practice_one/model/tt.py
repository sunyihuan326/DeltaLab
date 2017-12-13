# coding:utf-8
'''
Created on 2017/12/8.

@author: chk01
'''
import scipy.io as scio

# file = 'F:/dataSets/MNIST/mnist_data_small'
# data = scio.loadmat(file)
# from sklearn.model_selection import train_test_split
#
# print(data['X'].shape)
# print(data['Y'].shape)
# X_train, X_test, Y_train, Y_test = train_test_split(data['X'], data['Y'], test_size=0.2)
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)
import numpy as np
import tensorflow as tf

print(np.e)
print(-np.log(np.e / (np.e + 8)))

ZL = tf.Variable([[0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
print(ZL.shape)
Y = tf.constant([[0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=tf.float32)
# Y = tf.get_variable(dtype=tf.float32, shape=(1, 2), name='tt',initializer=tf.contrib.layers.xavier_initializer())
cor_op = tf.argmax(Y, 1)
pre_op = tf.argmax(ZL, 1)
cost1 = tf.square(tf.cast(cor_op - pre_op, dtype=tf.float32))
lost = tf.reduce_mean(
    cost1 + tf.nn.softmax_cross_entropy_with_logits(logits=ZL,
                                                    labels=Y))
# loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(lost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(30):
        sess.run(train_op)
        print(sess.run(lost))
        print(sess.run(tf.reduce_mean(cost1)))
        print(sess.run(tf.argmax(ZL, 1)))
        # 1.37195
        # 2.37195
