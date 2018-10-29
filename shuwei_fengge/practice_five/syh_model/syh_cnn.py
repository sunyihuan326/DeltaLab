# coding:utf-8 
'''
created on 2018/01/13

@author:sunyihuan
'''

import json

import numpy as np
import tensorflow as tf
from practice_five.utils import *
import matplotlib.pyplot as plt


# show data
# X_data = np.reshape(X_data, (-1, 28, 28))
def show_data(X, Y):
    for i in range(1, 10):
        plt.subplot(330 + i)
        plt.imshow(X[i] / 255)
        plt.title(Y[i])
    plt.show()


def preprocessing(trX, teX, trY, teY):
    # m, _ = trX.shape
    # bs = 2
    # res = SMOTE(ratio={0: int(bs * 0.23 * m), 1: int(bs * .54 * m), 2: int(bs * .23 * m)})
    # trX, trY = res.fit_sample(trX, np.argmax(trY, 1))
    # trY = np.eye(3)[trY]

    trX = trX / 255.
    teX = teX / 255.

    return trX, teX, trY, teY


def model(trX, trY, teX, teY, lr=.2, epoches=200, minibatch_size=64):
    m, features = trY.shape
    X = tf.placeholder(tf.float32, shape=[None, 64 * 64 * 3], name="input")

    XX = tf.reshape(X, shape=[-1, 64, 64, 3])
    inputs = XX
    Y = tf.placeholder(tf.float32, shape=[None, features])
    # YY = tf.one_hot(Y, 10, on_value=1, off_value=None, axis=1)

    # dp = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    # reg1 = tf.contrib.layers.l2_regularizer(scale=0.01)
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid')

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='same',
        activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid')
    # conv4 = tf.layers.conv2d(
    #     inputs=pool2,
    #     filters=64,
    #     kernel_size=[3, 3],
    #     strides=(1, 1),
    #     padding='same',
    #     activation=tf.nn.relu)
    #
    # conv5 = tf.layers.conv2d(
    #     inputs=conv4,
    #     filters=128,
    #     kernel_size=[3, 3],
    #     strides=(1, 1),
    #     padding='same',
    #     activation=tf.nn.relu)
    flatten = tf.layers.flatten(inputs=conv3)

    fc1 = tf.layers.dense(flatten, 16, activation=tf.nn.relu)
    # fc1 = tf.layers.batch_normalization(fc1)
    # fc1 = tf.layers.dropout(fc1, rate=dp, training=True)

    # fc2 = tf.layers.dense(fc1, 64, activation=tf.nn.relu)
    # fc2 = tf.layers.batch_normalization(fc2)
    # fc2 = tf.layers.dropout(fc2, rate=dp, training=True)

    # fc3 = tf.layers.dense(fc2, 4, activation=tf.nn.relu, name='fc3')
    # fc3 = tf.layers.batch_normalization(fc3)
    # fc3 = tf.layers.dropout(fc3, rate=dp, training=True)
    ZL = tf.layers.dense(fc1, features, activation=None, name='output')

    learning_rate = tf.train.exponential_decay(lr,
                                               global_step=global_step,
                                               decay_steps=100, decay_rate=0.9)
    learning_rate = tf.maximum(learning_rate, .001)
    loss = tf.losses.mean_squared_error(
        labels=Y, predictions=ZL)

    train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    # train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    add_global = global_step.assign_add(1)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoches):
            num_minibatches = int(m / minibatch_size)
            for minibatch_X, minibatch_Y in minibatches(trX, trY, minibatch_size, shuffle=True):
                __, _loss, _, res, llr = sess.run([add_global, loss, train_op, ZL, learning_rate],
                                                  feed_dict={X: minibatch_X, Y: minibatch_Y})
            print('epoch', epoch, 'loss', _loss)
            print(llr)

        zl = ZL.eval(feed_dict={X: teX})
        loss = loss.eval(feed_dict={X: teX, Y: teY})

        saver.save(sess, "save/model_conv3—3point-{}-{}.ckpt".format(epoches, int(loss)))
        # saver.restore(sess, "save/model.ckpt")
        print(loss)
    return zl


if __name__ == '__main__':
    file = '/Users/sunyihuan/Desktop/Data/face_top_9.mat'
    X_train, X_test, Y_train, Y_test = load_data(file)
    Y_train = Y_train[:, 6:12]
    Y_test = Y_test[:, 6:12]
    #
    zl = model(X_train, Y_train, X_test, Y_test, epoches=50)
    print(zl[0])
    print(zl[1])
    print(zl[100])

