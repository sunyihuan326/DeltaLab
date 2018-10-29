# coding:utf-8 
'''
created on 

@author:sunyihuan
'''

import tensorflow as tf
from tensorflow.python.framework import ops

from practice_one.model.utils import *
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, accuracy_score

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
    # res = RandomOverSampler(random_state=42)
    # trY = np.argmax(trY, 1)
    # teY = np.argmax(teY, 1)
    # trX, trY = res.fit_sample(trX, trY)
    # teX, teY = res.fit_sample(teX, teY)
    #
    # trY = np.eye(3)[trY]
    # teY = np.eye(3)[teY]
    return trX, teX, trY, teY


def create_placeholders(n_x, n_y):
    X = tf.placeholder(name='X', shape=(None, n_x), dtype=tf.float32)
    Y = tf.placeholder(name='Y', shape=(None, n_y), dtype=tf.float32)

    return X, Y


def forward_propagation(X, parameters, kp):
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A = tf.nn.relu(tf.add(tf.matmul(A_prev, tf.transpose(W)), b))
        A = tf.layers.batch_normalization(A, axis=-1)
        A = tf.nn.dropout(A, kp)
    ZL = tf.add(tf.matmul(A, tf.transpose(parameters['W' + str(L)])), parameters['b' + str(L)])
    return ZL


def model(X_train, Y_train, X_test, Y_test, layer_dims, keep_prob=1.0, epochs=2000, minibatch_size=64,
          initial_learning_rate=0.5, minest_learning_rate=0.01):
    ops.reset_default_graph()
    m, n_x = X_train.shape
    n_y = Y_train.shape[1]

    X, Y = create_placeholders(n_x, n_y)
    # X = tf.constant(X_train, dtype=tf.float32)
    # Y = tf.constant(Y_train, dtype=tf.float32)

    kp = tf.placeholder(tf.float32)

    parameters = scio.loadmat("face_1_channel_senseDNN_parameter.mat")

    a = tf.matmul(X, tf.transpose(parameters['W1']))
    ZL = tf.add(a, parameters['b1'])
    # ss = tf.where(tf.greater(abs(Y - ZL), 1), abs(Y - ZL) * 10, abs(Y - ZL) * 1)
    # cost = tf.reduce_sum(ss)
    correct_pred = tf.equal(tf.argmax(ZL, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train, kp: 1})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test, kp: 1})
        ZLte = ZL.eval({X: X_test, Y: Y_test, kp: 1})
        ZLtr = ZL.eval({X: X_train, Y: Y_train, kp: 1})
        ZYtr = list(np.argmax(ZLtr, 1))
        ZYte = list(np.argmax(ZLte, 1))
    for i in range(3):
        print(str(i) + "比例", round(100 * ZYte.count(i) / len(ZYte), 2), "%")

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    return ZYtr, ZYte


if __name__ == '__main__':
    name = 'Syh'
    if name == 'Dxq':
        file = 'F:/dataSets/MNIST/mnist_data_small.mat'
    elif name == 'Syh':
        file = 'face_1_channel_sense.mat'
    # load data
    X_train, X_test, Y_train, Y_test = load_data(file)

    # preprocessing
    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    data_check(Y_train)
    data_check(Y_test)

    layer_dims = [X_train.shape[1], Y_train.shape[1]]

    ztr, zte = model(X_train, Y_train, X_test, Y_test, layer_dims, keep_prob=0.99, epochs=10000,
                     initial_learning_rate=0.5)
    ctr = 0.
    cte = 0.
    # for i in range(len(z1)):
    #     if z1[i] not in accept_ans[np.argmax(Y_test, 1)[i]]:
    #         c += 1. / len(z1)
    # print(c)
    dl = {}
    dl["ztr"] = ztr
    dl["zte"] = zte
    scio.savemat("dnn_res_sense_undersample", dl)
    fpr, tpr, thresholds = roc_curve(y_true=np.argmax(Y_test, 1), y_score=zte, pos_label=2)
    print(fpr, tpr, thresholds)
    print(confusion_matrix(y_true=np.argmax(Y_test, 1), y_pred=zte))
    for i in range(len(zte)):
        if abs(zte[i] - np.argmax(Y_test, 1)[i]) > 1:
            cte += 1. / len(zte)
    print("cte", cte)

    for i in range(len(ztr)):
        if abs(ztr[i] - np.argmax(Y_train, 1)[i]) > 1:
            ctr += 1. / len(ztr)
    print("ctr", ctr)

    print(classification_report(y_pred=zte, y_true=np.argmax(Y_test, 1)))
