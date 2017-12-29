# coding:utf-8
'''
Created on 2017/12/8

@author: sunyihuan
'''

from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import matplotlib.pyplot as plt
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split
from practice_one.model.utils import *
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours
from sklearn.metrics import classification_report, confusion_matrix, hamming_loss, roc_curve, roc_auc_score


def preprocessing(trX, teX, trY, teY):
    trY = np.argmax(trY, 1)
    teY = np.argmax(teY, 1)
    trX, trY = SMOTE(ratio='auto', random_state=42).fit_sample(trX, trY)
    teX, teY = SMOTE(ratio='auto', random_state=42).fit_sample(teX, teY)

    trY = np.eye(3)[trY]
    teY = np.eye(3)[teY]
    return trX, teX, trY, teY


def data_check(data):
    res = list(np.argmax(data.T, 1))
    num = len(res)
    classes = data.shape[0]
    for i in range(classes):
        print(str(i) + '的比例', round(100.0 * res.count(i) / num, 2), '%')
    print('<------------------分割线---------------------->')


def main(Xtr, Ytr, Xte, Yte, k):
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])
    parameters = {}

    kmeans = KMeans(inputs=X, num_clusters=k,
                    distance_metric='cosine',
                    use_mini_batch=True)

    res = kmeans.training_graph()
    (all_scores, cluster_idx, scores, cluster_centers_initialized, cluster_centers_var, init_op,
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
    counts = np.zeros(shape=(k, num_classes))
    for i in range(len(idx)):
        counts[idx[i]] += Ytr[i]
    # Assign the most frequent label to the centroid
    labels_map = [np.argmax(c) for c in counts]

    parameters['labels_map'] = labels_map
    # scio.savemat('kmeans_parameters', parameters)
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
    with sess.as_default():
        accuracy_train_op = accuracy_op.eval(feed_dict={X: Xtr, Y: Ytr})
        accuracy_test_op = accuracy_op.eval(feed_dict={X: test_x, Y: test_y})
        cluster_label = cluster_label.eval(feed_dict={X: test_x})
        print("Train Accuracy:", accuracy_train_op)
        print("Test Accuracy:", accuracy_test_op)
    # print('correct_prediction', correct_prediction)

    return cluster_label


def load_data_train(file):
    data_train = scio.loadmat(file_sense)
    data_X = np.reshape(data_train['X'], [data_train['X'].shape[0], num_features])
    X_train = data_X[:1200, :] / 255.
    Y_train = data_train['Y'][:1200, :]

    X_test = data_X[1200:, :] / 255
    Y_test = data_train['Y'][1200:, :]
    return X_train, Y_train, X_test, Y_test


num_steps = 500  # Total steps to train
batch_size = 1024  # The number of samples per batch
k = 300  # The number of clusters
num_classes = 3  # The 10 digits
num_features = 28  # Each image is 28x28 pixels

if __name__ == '__main__':

    name = 'Syh'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY'
    elif name == 'Syh':
        file_sense = 'face_1_channel_sense.mat'

    # 样本比例检测
    # data_check(Y_test)
    # data_check(Y_train)

    X_train0, X_test0, Y_train0, Y_test0 = load_data(file_sense)
    X_train0, X_test0, Y_train0, Y_test0 = preprocessing(X_train0, X_test0, Y_train0, Y_test0)

    # X_test0 = X_test0.reshape(-1, X_test0.shape[1] * X_test0.shape[2])
    cluster_sense = main(X_train0, Y_train0, X_test0, Y_test0, k=k)

    # cl = {}
    # cl["sense"] = cluster_sense
    # scio.savemat("sense_cluster", cl)

    for i in range(3):
        print(str(i) + '的比例', round(100.0 * list(cluster_sense).count(i) / len(cluster_sense), 2), '%')

    print("#################")
    Y_train0 = list(np.argmax(Y_train0, 1))
    for i in range(3):
        print(str(i) + '的比例', round(100.0 * list(Y_train0).count(i) / len(Y_train0), 2), '%')

    # 计算有多少小判断成大
    Y_test0 = list(np.argmax(Y_test0, 1))
    c = 0
    for i in range(len(cluster_sense)):
        if abs(cluster_sense[i] - Y_test0[i]) > 1:
            c = c + 1
    print(round(100 * (c / len(cluster_sense)), 2), "%")

    print(classification_report(y_pred=cluster_sense, y_true=Y_test0))

    print(confusion_matrix(y_pred=cluster_sense, y_true=Y_test0))
    print(hamming_loss(y_pred=cluster_sense, y_true=Y_test0))

    fpr, tpr, thresholds = roc_curve(y_true=Y_test0, y_score=cluster_sense, pos_label=2)

    print("fpr", fpr)
    print("tpr", tpr)
    # roc_auc = roc_auc_score(fpr, tpr,average="macro")
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
    plt.plot(fpr, tpr, 'r')
    plt.show()

    # cluster_sense = pd.DataFrame(cluster_sense)
    # cluster_sense.to_csv('cluster_outlin.csv')
