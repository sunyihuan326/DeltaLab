# coding:utf-8
'''
Created on 2017/12/6

@author: sunyihuan
'''
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# Ignore all GPUs, tf random forest does not benefit from it.
import os
import scipy.io as scio

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(Xtr, Ytr, Xte, Yte):
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])

    kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                    use_mini_batch=True)

    res = kmeans.training_graph()
    (all_scores, cluster_idx, scores, cluster_centers_initialized, cluster_centers_var, init_op,
     train_op) = kmeans.training_graph()
    cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)

    # Initialize the variables (i.e. assign their default value)
    init_vars = tf.global_variables_initializer()

    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars, feed_dict={X: Xtr})
    sess.run(init_op, feed_dict={X: Xtr})

    # Training
    for i in range(1, num_steps + 1):
        _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                             feed_dict={X: Xtr})
        if i % 10 == 0 or i == 1:
            print("Step %i, Avg Distance: %f" % (i, d))

    # Assign a label to each centroid
    # Count total number of labels per centroid, using the label of each training
    # sample to their closest centroid (given by 'idx')
    counts = np.zeros(shape=(k, num_classes))
    for i in range(len(idx)):
        counts[idx[i]] += Ytr[i]
    # Assign the most frequent label to the centroid
    labels_map = [np.argmax(c) for c in counts]
    labels_map = tf.convert_to_tensor(labels_map)

    # Evaluation ops
    # Lookup: centroid_id -> label
    cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
    # Compute accuracy
    correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Test Model
    test_x, test_y = Xte, Yte
    print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))


num_steps = 500  # Total steps to train
batch_size = 1024  # The number of samples per batch
k = 25  # The number of clusters
num_classes = 9  # The 10 digits
num_features = 128 * 128  # Each image is 28x28 pixels
if __name__ == '__main__':
    name = 'Dxq'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY'
    elif name == 'Syh':
        file = 'E:/deeplearning_Data/face_1_channel_XY'

    data_train = scio.loadmat(file + 'DNN2_train')
    X_train = data_train['X'] / 255.
    Y_train = data_train['Y']

    data_test = scio.loadmat(file + 'DNN2_test')
    X_test = data_test['X'] / 255.
    Y_test = data_test['Y']
    # Parameters
    main(X_train.T, Y_train.T, X_test.T, Y_test.T)
