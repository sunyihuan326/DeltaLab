# coding:utf-8
'''
Created on 2017/11/15.

@author: chk01
'''
import tensorflow as tf
from tensorflow.python.framework import ops
from practice_one.model.utils import *


def preprocessing(trX, teX, trY, teY):
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


def accuracy_cal(train_pre_val, train_cor_val):
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
    error_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, len(train_cor_val), 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct = 0
    real_correct = 0
    pre_real = np.zeros([len(train_cor_val), 2])
    for i in range(len(train_cor_val)):
        pre_real[i, :] = [train_pre_val[i], train_cor_val[i]]
        error_count[train_cor_val[i] + 10] += 1
        if train_pre_val[i] in accept_ans[train_cor_val[i]]:
            correct += 1
        else:
            error_count[train_cor_val[i]] += 1
        if train_pre_val[i] == train_cor_val[i]:
            real_correct += 1
    scio.savemat('result_analysis/64DNN2/res', {'result': pre_real})
    scio.savemat('result_analysis/64DNN2/error', {'result': error_count})
    return correct / len(train_cor_val), real_correct / len(train_cor_val)


def model(X_train, Y_train, X_test, Y_test, layer_dims, keep_prob=1.0, epochs=2000, minibatch_size=64,
          initial_learning_rate=0.5, minest_learning_rate=0.01):
    ops.reset_default_graph()

    m, n_x = X_train.shape
    n_y = Y_train.shape[1]

    kp = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters_deep(layer_dims)

    ZL = forward_propagation(X, parameters, keep_prob)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ZL, labels=Y))
    tf.summary.scalar(name='cost', tensor=cost)

    predict_op = tf.argmax(ZL, 1)
    correct_op = tf.argmax(Y, 1)

    correct_pred = tf.equal(predict_op, correct_op)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar(name='accuracy', tensor=accuracy)
    tf.summary.histogram(name='predict_op', values=predict_op)
    tf.summary.histogram(name='correct_op', values=correct_op)

    # cost = compute_cost(Z1, Y) + tf.contrib.layers.l1_regularizer(.2)(parameters['W1'])
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.99).minimize(cost)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10, decay_rate=0.9)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate + 0.01).minimize(cost)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    add_global = global_step.assign_add(1)
    init = tf.global_variables_initializer()

    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)

        summary = tf.summary.FileWriter(logdir='logdir/64DNN2', graph=sess.graph)
        for epoch in range(epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                acc, summary_op, zl, par, _, temp_cost, _ = sess.run(
                    [accuracy, merged_summary_op, ZL, parameters, optimizer, cost, add_global],
                    feed_dict={X: minibatch_X, Y: minibatch_Y, kp: keep_prob})
                summary.add_summary(summary_op, epoch)
                minibatch_cost += temp_cost / num_minibatches

            if epoch % 50 == 0:
                print("Cost|Acc after epoch %i: %f | %f" % (epoch, temp_cost, acc))

                train_pre_val = predict_op.eval({X: X_train, Y: Y_train, kp: 1})
                train_cor_val = correct_op.eval({X: X_train, Y: Y_train, kp: 1})
                train_accuracy, train_real_accuracy = accuracy_cal(train_pre_val, train_cor_val)

                test_pre_val = predict_op.eval({X: X_test, Y: Y_test, kp: 1})
                test_cor_val = correct_op.eval({X: X_test, Y: Y_test, kp: 1})
                test_accuracy, test_real_accuracy = accuracy_cal(test_pre_val, test_cor_val)

                print("Train Accuracy:", train_accuracy, train_real_accuracy)
                print("Test Accuracy:", test_accuracy, test_real_accuracy)
                print("----------------------------------------------------------")

        train_pre_val = predict_op.eval({X: X_train, Y: Y_train, kp: 1})
        train_cor_val = correct_op.eval({X: X_train, Y: Y_train, kp: 1})
        train_accuracy, train_real_accuracy = accuracy_cal(train_pre_val, train_cor_val)

        test_pre_val = predict_op.eval({X: X_test, Y: Y_test, kp: 1})
        test_cor_val = correct_op.eval({X: X_test, Y: Y_test, kp: 1})
        test_accuracy, test_real_accuracy = accuracy_cal(test_pre_val, test_cor_val)

        print("Train Accuracy:", train_accuracy, train_real_accuracy)
        print("Test Accuracy:", test_accuracy, test_real_accuracy)
        print("----------------------------------------------------------")

    return par


if __name__ == '__main__':
    name = 'Dxq'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel1/face_1_channel_XY64'
    elif name == 'Syh':
        file = 'E:/deeplearning_Data/face_1_channel_XY'

    # load data
    X_train, X_test, Y_train, Y_test = load_data(file, test_size=0.2)
    # preprocessing
    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)
    data_check(Y_train)
    data_check(Y_test)

    layer_dims = [X_train.shape[1], Y_train.shape[1]]
    data_check(Y_test)
    data_check(Y_train)

    parameters = model(X_train, Y_train, X_test, Y_test, layer_dims, keep_prob=0.7, epochs=200, initial_learning_rate=0.5)

    scio.savemat(file + '64DNN2_parameter', parameters)
