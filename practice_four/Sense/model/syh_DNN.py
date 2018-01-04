# coding:utf-8 
'''
created on 

@author:sunyihuan
'''

import tensorflow as tf
from tensorflow.python.framework import ops

from practice_one.model.utils import *
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, RepeatedEditedNearestNeighbours
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, accuracy_score

absolute_error = [
    [2, 5, 6, 7, 8],
    [6, 7, 8],
    [0, 3, 6, 7, 8],
    [2, 5, 8],
    [0, 2, 6, 8],
    [0, 3, 6],
    [0, 1, 2, 5, 8],
    [0, 1, 2],
    [0, 1, 2, 3, 6]
]


def preprocessing(trX, teX, trY, teY):
    res = RandomOverSampler(ratio="all")
    trX, trY = res.fit_sample(trX, np.argmax(trY, 1))
    trY = np.eye(3)[trY]
    return trX, teX, trY, teY


def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = tf.get_variable(dtype=tf.float32, name='W' + str(l),
                                                   shape=(layer_dims[l], layer_dims[l - 1]),
                                                   initializer=tf.contrib.layers.xavier_initializer())
        parameters['b' + str(l)] = tf.get_variable(dtype=tf.float32, name='b' + str(l),
                                                   shape=(1, layer_dims[l]),
                                                   initializer=tf.contrib.layers.xavier_initializer())
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (1, layer_dims[l]))

    return parameters


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
          initial_learning_rate=0.5, minest_learning_rate=0.001):
    ops.reset_default_graph()

    m, n_x = X_train.shape
    n_y = Y_train.shape[1]

    kp = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters_deep(layer_dims)

    ZL = forward_propagation(X, parameters, kp)
    # ss = tf.where(tf.greater(abs(Y - ZL), 1), abs(Y - ZL) * 10, abs(Y - ZL) * 1)
    # cost = tf.reduce_sum(ss)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ZL, labels=Y))
    tf.summary.scalar('cost', cost)
    # cost = compute_cost(Z1, Y) + tf.contrib.layers.l1_regularizer(.2)(parameters['W1'])
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.99).minimize(cost)

    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=1000, decay_rate=0.95)
    learning_rate = tf.maximum(learning_rate, minest_learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    add_global = global_step.assign_add(1)
    merge_op = tf.summary.merge_all()

    correct_pred = tf.equal(tf.argmax(ZL, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(logdir='logdir/MNIST/DNN')
        for epoch in range(epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                summary, zl, par, _, temp_cost, _, acc = sess.run(
                    [merge_op, ZL, parameters, optimizer, cost, add_global, accuracy],
                    feed_dict={X: minibatch_X, Y: minibatch_Y, kp: keep_prob})
                minibatch_cost += temp_cost / num_minibatches
                writer.add_summary(summary)
            if epoch % 200 == 0:
                print("Cost|Acc after epoch %i: %f | %f" % (epoch, temp_cost, acc))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train, kp: 1})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test, kp: 1})
        ZL = ZL.eval({X: X_test, Y: Y_test, kp: 1})
        ZY = list(np.argmax(ZL, 1))

        # predict data distribution for 3 classes
        for i in range(3):
            print(str(i) + "比例", round(100 * ZY.count(i) / len(ZY), 2), "%")

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

    return par, ZY


if __name__ == '__main__':
    file = 'sense64x64.mat'
    # load data
    X_train, X_test, Y_train, Y_test = load_data(file, test_size=0.2)

    # preprocessing
    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)

    # data distribution
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    data_check(Y_train)
    data_check(Y_test)

    layer_dims = [X_train.shape[1], Y_train.shape[1]]

    parameters, z1 = model(X_train, Y_train, X_test, Y_test, layer_dims, keep_prob=0.9, epochs=1000,
                           initial_learning_rate=0.5)

    # caculate accept accuracy
    c = 0.
    for i in range(len(z1)):
        if abs(z1[i] - np.argmax(Y_test, 1)[i]) > 1:
            c += 1 / len(z1)
    print("accept error", c)

    # roc curve
    fpr, tpr, thresholds = roc_curve(y_true=np.argmax(Y_test, 1), y_score=z1, pos_label=2)
    # confusion matrix
    print(confusion_matrix(y_true=np.argmax(Y_test, 1), y_pred=z1))

    # Classification index
    print(classification_report(y_pred=z1, y_true=np.argmax(Y_test, 1)))
    scio.savemat(file + 'DNN_parameter', parameters)
