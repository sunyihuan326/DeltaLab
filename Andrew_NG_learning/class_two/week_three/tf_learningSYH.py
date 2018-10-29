# coding:utf-8
'''
Created on 2017/11/14

@author: sunyihuan
'''

import numpy as np
import tensorflow as tf

coefficients = np.array([[1.], [-10.], [25.]])

print(coefficients.shape)
w = tf.Variable([0], dtype=tf.float32)
# x = tf.placeholder(tf.float32, shape=[3, 1])
x = tf.placeholder(tf.float32,[3,1])
# x= tf.constant(coefficients,tf.float32)
cost = x[0][0] * w ** 2 + x[1][0] * w + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
tra=tf.train.AdamOptimizer()
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))

for i in range(1000):
    session.run(train,feed_dict={x:coefficients})

print(session.run(w))
