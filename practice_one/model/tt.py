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
import scipy.io as scio
import tensorflow as tf
from practice_one.model.utils import *
from tensorflow.contrib.factorization import KMeans

# print(np.e)
# print(-np.log(np.e / (np.e + 8)))

# ZL = tf.Variable([[0, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
# print(ZL.shape)
# Y = tf.constant([[0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=tf.float32)
# Y = tf.get_variable(dtype=tf.float32, shape=(1, 2), name='tt',initializer=tf.contrib.layers.xavier_initializer())
# cor_op = tf.argmax(Y, 1)
# pre_op = tf.argmax(ZL, 1)
# cost1 = tf.square(tf.cast(cor_op - pre_op, dtype=tf.float32))
# lost = tf.reduce_mean(
#     cost1 + tf.nn.softmax_cross_entropy_with_logits(logits=ZL,
#                                                     labels=Y))
# # loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(lost)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(30):
#         sess.run(train_op)
#         print(sess.run(lost))
#         print(sess.run(tf.reduce_mean(cost1)))
#         print(sess.run(tf.argmax(ZL, 1)))
#         1.37195
#           2.37195
parameters = scio.loadmat('kmeans_parameters.mat')
X_train, X_test, Y_train, Y_test = load_data("face_1_channel_sense.mat")
print(X_test.shape)
num_features = 28
num_classes = 3

X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

kmeans = KMeans(inputs=X, num_clusters=300,
                distance_metric='cosine',
                use_mini_batch=True)

(all_scores, cluster_idx, scores, cluster_centers_initialized, cluster_centers_var, init_op,
 train_op) = kmeans.training_graph()
cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

sess.run(init_vars, feed_dict={X: X_test})
sess.run(init_op, feed_dict={X: X_test})
cl = sess.run(cluster_idx, feed_dict={X: X_test})
print(cl)

parameters = scio.loadmat('kmeans_parameters.mat')
labels_map = tf.convert_to_tensor(parameters['labels_map'][0])

# Evaluation ops
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)

# Test Model
test_x, test_y = X_test, Y_test
with sess.as_default():
    cluster_label = cluster_label.eval(feed_dict={X: X_test})

c = 0
for i in range(len(cluster_label)):
    if abs(cluster_label[i] - np.argmax(Y_train, 1)[i]) > 1:
        c += 1. / len(cluster_label)
print(c)

tt = scio.loadmat("tt_cluster_label.mat")
sense = scio.loadmat("sense_cluster.mat")
tt = tt["tt"][0]
se = sense["sense"][0]
for i in range(len(tt)):
    if tt[i] != se[i]:
        print(i, tt[i], se[i])

# print('correct_prediction', correct_prediction)
