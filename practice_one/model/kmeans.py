# coding:utf-8
'''
Created on 2017/12/6

@author: sunyihuan
'''
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# Ignore all GPUs, tf random forest does not benefit from it.
import os
import scipy.io as scio
import numpy as np
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def data_check(data):
    res = list(np.argmax(data.T, 1))
    num = len(res)
    classes = data.shape[0]
    for i in range(classes):
        print(str(i) + '的比例', round(100.0 * res.count(i) / num, 2), '%')
    print('<------------------分割线---------------------->')


def main(Xtr, Ytr, Xte, Yte):
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])

    kmeans = KMeans(inputs=X, num_clusters=k,
                    distance_metric='cosine',
                    use_mini_batch=True)

    res = kmeans.training_graph()
    (all_scores, cluster_idx, scores, cluster_centers_initialized, init_op,
     train_op) = kmeans.training_graph()
    cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)
    tf.summary.scalar(name='avg_distance', tensor=avg_distance)
    tf.summary.histogram('cluster_idx', cluster_idx)
    merge_all_op = tf.summary.merge_all()

    # Initialize the variables (i.e. assign their default value)
    init_vars = tf.global_variables_initializer()

    # Start TensorFlow session
    sess = tf.Session()
    write = tf.summary.FileWriter('logdir/KMeans', sess.graph)

    # Run the initializer
    sess.run(init_vars, feed_dict={X: Xtr})
    sess.run(init_op, feed_dict={X: Xtr})

    # Training
    for i in range(1, num_steps + 1):
        summary_write, _, d, idx = sess.run(
            [merge_all_op, train_op, avg_distance, cluster_idx],
            feed_dict={X: Xtr})
        if i % 10 == 0 or i == 1:
            write.add_summary(summary_write, i)
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
    # print(correct_prediction)
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Test Model
    test_x, test_y = Xte, Yte
    Y, cluster_label, correct_prediction, accuracy_op = sess.run(
        [tf.argmax(Y, 1), cluster_label, correct_prediction, accuracy_op],
        feed_dict={X: test_x, Y: test_y})
    print("Test Accuracy:", accuracy_op)
    # print('correct_prediction', correct_prediction)

    # 计算有多少小判断成大
    c = 0
    for i in range(len(correct_prediction)):
        if correct_prediction[i] == False:
            if abs(cluster_label[i] - Y[i]) > 1:
                c += 1
    print(round(100 * (c / test_y.shape[0]), 2), "%")


num_steps = 30  # Total steps to train
batch_size = 1024  # The number of samples per batch
k = 22  # The number of clusters
num_classes = 9  # The 10 digits
num_features = 128 * 128  # Each image is 28x28 pixels

if __name__ == '__main__':

    name = 'Syh'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY'
    elif name == 'Syh':
        file = 'E:/deeplearning_Data/face_1_channel_XY'

    data_train = scio.loadmat(file)
    X_train, X_test, Y_train, Y_test = train_test_split(data_train['X'], data_train['Y'], train_size=0.8)
    X_train = X_train / 255.
    X_test = X_test / 255.

    # Parameters
    main(X_train.reshape(-1,128*128), Y_train, X_test.reshape(-1,128*128), Y_test)