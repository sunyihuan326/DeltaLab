# coding:utf-8
'''
Created on 2017/11/14

@author: sunyihuan
'''

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from class_two.week_three.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)

y_hat = tf.constant(36, name='y_hat')  # Define y_hat constant. Set to 36.
y = tf.constant(39, name='y')  # Define y. Set to 39
with tf.Session() as session:
    session.run(y_hat)
print(session.run(y_hat))
