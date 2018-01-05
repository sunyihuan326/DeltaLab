# coding:utf-8
'''
Created on 2017/11/15.

@author: chk01
'''
import tensorflow as tf
from tensorflow.python.framework import ops
from practice_four.utils import *
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import confusion_matrix, classification_report


def preprocessing(trX, teX, trY, teY):
    # res = SMOTE(ratio="auto")
    # trX, trY = res.fit_sample(trX, np.argmax(trY, 1))
    # trY = np.eye(3)[trY]
    #
    trX = trX / 255.
    teX = teX / 255.

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
    # A = tf.layers.batch_normalization(A, axis=-1)
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
          initial_learning_rate=0.5, minest_learning_rate=0.0001):
    ops.reset_default_graph()

    m, n_x = X_train.shape
    n_y = Y_train.shape[1]

    kp = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters_deep(layer_dims)

    ZL = forward_propagation(X, parameters, keep_prob)

    predict_op = tf.argmax(ZL, 1)
    correct_op = tf.argmax(Y, 1)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ZL, labels=Y))
    wcost = tf.contrib.layers.l2_regularizer(.1)(parameters['W1'])

    # cost = cost + wcost
    # cost = tf.reduce_mean(tf.square(ZL - Y))
    tf.summary.scalar(name='cost', tensor=cost)

    correct_pred = tf.equal(predict_op, correct_op)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar(name='accuracy', tensor=accuracy)
    tf.summary.histogram(name='predict_op', values=predict_op)
    tf.summary.histogram(name='correct_op', values=correct_op)

    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10, decay_rate=0.9)
    learning_rate = tf.maximum(learning_rate, minest_learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99).minimize(cost)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    add_global = global_step.assign_add(1)
    init = tf.global_variables_initializer()

    merged_summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)

        summary = tf.summary.FileWriter(logdir='logdir', graph=sess.graph)
        for epoch in range(epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)

            for minibatch_X, minibatch_Y in minibatches(X_train, Y_train, minibatch_size, shuffle=True):
                wwc, acc, summary_op, zl, par, _, temp_cost, _ = sess.run(
                    [wcost, accuracy, merged_summary_op, ZL, parameters, optimizer, cost, add_global],
                    feed_dict={X: minibatch_X, Y: minibatch_Y, kp: keep_prob})
                summary.add_summary(summary_op, epoch)
                minibatch_cost += temp_cost / num_minibatches

            if epoch % 50 == 0:
                print("Cost|Acc after epoch %i: %f" % (epoch, minibatch_cost))
                train_pre_val = accuracy.eval({X: X_train_org, Y: Y_train_org, kp: 1})
                test_pre_val = accuracy.eval({X: X_test, Y: Y_test, kp: 1})
                print("Train Accuracy:", train_pre_val)
                print("Test Accuracy:", test_pre_val)
                # print("wcost", wwc)
            if epoch % 200 == 10:
                test_pre_val = predict_op.eval({X: X_test, Y: Y_test, kp: 1})
                test_res_matrix = confusion_matrix(y_true=np.argmax(Y_test, 1), y_pred=test_pre_val)
                accuracy_cal(test_res_matrix, 'test')
            if epoch % 500 == 10:
                train_pre_val = predict_op.eval({X: X_train_org, Y: Y_train_org, kp: 1})
                train_res_matrix = confusion_matrix(y_true=np.argmax(Y_train_org, 1), y_pred=train_pre_val)
                accuracy_cal(train_res_matrix, 'train')



        print('↓↓↓↓↓↓↓↓↓↓↓--------结果------------↓↓↓↓↓↓↓↓↓↓↓↓↓↓')
        train_pre_val = predict_op.eval({X: X_train_org, Y: Y_train_org, kp: 1})
        train_res_matrix = confusion_matrix(y_true=np.argmax(Y_train_org, 1), y_pred=train_pre_val)
        accuracy_cal(train_res_matrix, 'train')
        test_pre_val = predict_op.eval({X: X_test, Y: Y_test, kp: 1})
        test_res_matrix = confusion_matrix(y_true=np.argmax(Y_test, 1), y_pred=test_pre_val)
        accuracy_cal(test_res_matrix, 'test')

        print('--------------Test-----------\n', classification_report(y_pred=test_pre_val, y_true=np.argmax(Y_test, 1)))
        print('--------------Train-----------\n', classification_report(y_pred=train_pre_val, y_true=np.argmax(Y_train_org, 1)))
    return par


if __name__ == '__main__':
    file = '../data/outline64x64.mat'
    # load data
    X_train_org, X_test_org, Y_train_org, Y_test_org = load_data(file, test_size=0.2)
    # preprocessing
    X_train, X_test, Y_train, Y_test = preprocessing(X_train_org, X_test_org, Y_train_org, Y_test_org)
    data_check(Y_train)
    data_check(Y_test)

    layer_dims = [X_train.shape[1], Y_train.shape[1]]
    epochs = 255
    parameters = model(X_train, Y_train, X_test, Y_test, layer_dims, keep_prob=.98, epochs=epochs,
                       initial_learning_rate=0.5)

    scio.savemat('parameter/outline64x64_parameter-{}'.format(epochs), parameters)
