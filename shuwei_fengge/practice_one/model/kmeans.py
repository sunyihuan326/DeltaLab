# coding:utf-8
'''
Created on 2017/12/6

@author: sunyihuan
'''
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.factorization import KMeans

# Ignore all GPUs, tf random forest does not benefit from it.
import os
from practice_one.model.utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


def preprocessing(trX, teX, trY, teY):
    return trX / 255., teX / 255., trY, teY


def create_placeholders(n_x, n_y):
    X = tf.placeholder(name='X', shape=(None, n_x), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(None, n_y), dtype=tf.float32)

    return X, Y


def main(Xtr, Ytr, Xte, Yte):
    ops.reset_default_graph()
    n_x = X_train.shape[1]
    n_y = Y_train.shape[1]

    X, Y = create_placeholders(n_x, n_y)

    kmeans = KMeans(inputs=X, num_clusters=k,
                    distance_metric='cosine',
                    use_mini_batch=True,
                    mini_batch_steps_per_iteration=8, )

    kmeans.training_graph()
    (all_scores, cluster_idx, scores, cluster_centers_initialized, init_op,
     train_op) = kmeans.training_graph()
    cluster_idx = cluster_idx[0]  # fix for cluster_idx being a tuple
    avg_distance = tf.reduce_mean(scores)

    accept_train_accuracy = 0.
    accept_test_accuracy = 0.

    tf.summary.scalar(name='avg_distance', tensor=avg_distance)
    tf.summary.histogram('cluster_idx', cluster_idx)
    with tf.name_scope('accuracy'):
        tf.summary.scalar('name=accept_train_accuracy', accept_train_accuracy)
        tf.summary.scalar('name=accept_test_accuracy', accept_test_accuracy)
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
        if i % 50 == 0 or i == 1:
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
    #
    # with sess.as_default():
    #     accuracy_train_op = accuracy_op.eval(feed_dict={X: Xtr, Y: Ytr})
    #     cluster_train_label = cluster_label.eval(feed_dict={X: Xtr, Y: Ytr})
    #     accuracy_test_op = accuracy_op.eval(feed_dict={X: Xte, Y: Yte})
    #     cluster_test_label = cluster_label.eval(feed_dict={X: Xte, Y: Yte})
    #     print("Train Accuracy:", accuracy_train_op)
    #     print("Test Accuracy:", accuracy_test_op)

    # Test Model
    test_x, test_y = Xte, Yte
    Y_tr, cluster_label, correct_prediction, accuracy_train_op = sess.run(
        [tf.argmax(Y, 1), cluster_label, correct_prediction, accuracy_op],
        feed_dict={X: Xtr, Y: Ytr})
    Y_te, accuracy_test_op = sess.run([tf.argmax(Y, 1), accuracy_op], feed_dict={X: test_x, Y: test_y})
    for i in range(len(Y_tr)):
        if Y_tr[i] - np.argmax(Y_train[i]) > 1:
            accept_train_accuracy += 1. / len(Y_tr)
    for j in range(len(Y_te)):
        if Y_te[j] - np.argmax(Y_test[j]) > 1:
            accept_test_accuracy += 1. / len(Y_te)
    print("Train Accuracy:", accuracy_train_op)
    print("Train  error:", accept_train_accuracy)
    print("Test Accuracy:", accuracy_test_op)
    print("Test  error:", accept_test_accuracy)
    # Y_tr = np.argmax(Ytr, 1)
    # for i in range(len(Y_tr)):
    #     if cluster_train_label[i] in accept_ans[Y_tr[i]]:
    #         accept_train_accuracy += 1. / len(Y_tr)
    # print("Train accept Accuracy:", accept_train_accuracy)

    return correct_prediction, cluster_label


def error_id(correct_prediction, Y, cluster_label=None):
    error_id = []
    for i in range(len(correct_prediction)):
        if cluster_label[i] not in accept_ans[Y[i]]:
            error_id.append(i)
            error_id.append(cluster_label[i])
            error_id.append(Y[i])
    return error_id


if __name__ == '__main__':
    num_steps = 50  # Total steps to train
    batch_size = 32  # The number of samples per batch
    k = 900  # The number of clusters
    num_classes = 3  # The 10 digits

    name = 'Syh'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY64'
    elif name == 'Syh':
        file = 'E:/deeplearning_Data/face_1_channel_outline13'

    X_train, X_test, Y_train, Y_test = load_data(file, test_size=0.3)
    print(X_train.shape)

    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)
    # Parameters
    correct_prediction, cluster_label = main(X_train, Y_train, X_test, Y_test)
    #
    # for i in range(9):
    #     print(str(i) + '的比例', round(100.0 * list(cluster_label).count(i) / len(cluster_label), 2), '%')
    #
    # print("++++++++++++++++++++++++++++++")

    data_check(Y_train)

    # Y_id = np.argmax(Y_test, 1)
    # print(error_id(correct_prediction, Y_id, cluster_label=cluster_label))
