# coding:utf-8
'''
Created on 2018/1/6.

@author: chk01
'''

from practice_five.utils import *
import matplotlib.pyplot as plt
import tensorflow as tf


# show data
# X_data = np.reshape(X_data, (-1, 28, 28))
def show_data(X, Y):
    X = np.reshape(X, (-1, 64, 64))
    for i in range(1, 10):
        plt.subplot(330 + i)
        plt.imshow(X[i] / 255, cmap='gray')
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


def model(trX, trY, teX, teY, lr=5.0, epoches=200, minibatch_size=64):
    m, features = trY.shape
    X = tf.placeholder(tf.float32, shape=[None, 64 * 64])
    XX = tf.reshape(X, shape=[-1, 64, 64, 1])
    Y = tf.placeholder(tf.float32, shape=[None, features])
    # YY = tf.one_hot(Y, 10, on_value=1, off_value=None, axis=1)

    # dp = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    reg1 = tf.contrib.layers.l2_regularizer(scale=0.1)
    conv1 = tf.layers.conv2d(XX, 24, 5, padding='same', activation=tf.nn.relu, kernel_regularizer=reg1)
    # conv1 = tf.layers.conv2d(XX, 24, 5, padding='same', activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same')

    conv2 = tf.layers.conv2d(conv1, 36, 5, padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same')

    conv3 = tf.layers.conv2d(conv2, 48, 5, padding='same', activation=tf.nn.relu)
    conv3 = tf.layers.max_pooling2d(conv3, 2, 2, padding='same')
    #
    # conv4 = tf.layers.conv2d(conv3, 64, 3, padding='same', activation=tf.nn.relu)
    # conv4 = tf.layers.max_pooling2d(conv4, 2, 2, padding='same')
    #
    # conv5 = tf.layers.conv2d(conv4, 64, 3, padding='same', activation=tf.nn.relu)
    # conv5 = tf.layers.max_pooling2d(conv5, 2, 2, padding='same')

    # convZ = tf.layers.flatten(pool3)
    convZ = tf.contrib.layers.flatten(conv3)

    fc1 = tf.layers.dense(convZ, 2 * features, activation=tf.nn.relu)
    fc1 = tf.layers.batch_normalization(fc1)
    # fc1 = tf.layers.dropout(fc1, rate=dp, training=True)

    # fc2 = tf.layers.dense(fc1, 2 * features, activation=tf.nn.relu)
    # fc2 = tf.layers.batch_normalization(fc2)
    # fc2 = tf.layers.dropout(fc2, rate=dp, training=True)
    #

    ZL = tf.layers.dense(fc1, features, activation=None, name='output')
    print(ZL)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ZL, labels=Y))
    # learning_rate = tf.train.piecewise_constant(global_step, [100, 200, 300, 600, 1000],
    #                                             [5.0, 2.5, 1.25, 0.5, 0.1, 0.01], name=None)
    # learning_rate = tf.train.exponential_decay(lr,
    #                                            global_step=global_step,
    #                                            decay_steps=100, decay_rate=0.9, staircase=True)
    # learning_rate = tf.train.polynomial_decay(lr, global_step, decay_steps=100,
    #                                           end_learning_rate=0.0001, power=1.0,
    #                                           cycle=True, name=None)
    # learning_rate = tf.train.natural_exp_decay(learning_rate, global_step, decay_steps=100, decay_rate=0.9,
    #                                            staircase=False, name=None)
    learning_rate = tf.train.inverse_time_decay(lr, global_step, decay_steps=30, decay_rate=0.9,
                                                staircase=True, name=None)
    # 自然数指数下降比
    # exponential_decay
    # 要快许多，适用于较快收敛，容易训练的网络。
    learning_rate = tf.maximum(learning_rate, .0001)
    loss = tf.reduce_mean(tf.square(Y - ZL) * [1, 2, 1, 2])

    train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)
    # train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    add_global = global_step.assign_add(1)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoches):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            for minibatch_X, minibatch_Y in minibatches(trX, trY, minibatch_size, shuffle=True):
                __, _loss, _, res, llr = sess.run([add_global, loss, train_op, ZL, learning_rate],
                                                  feed_dict={X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += _loss / num_minibatches

            if epoch % 10 == 0:
                test_loss = loss.eval(feed_dict={X: teX, Y: teY})
                print('epoch', epoch, 'test_loss', test_loss, 'train_loss', minibatch_cost)
                print(llr)

        saver.save(sess, "save/model-{}-{}.ckpt".format(epoches, int(test_loss)))


if __name__ == '__main__':
    file = '../data/face_top_9_L.mat'

    # load data
    X_train_org, X_test_org, Y_train_org, Y_test_org = load_data(file, test_size=0.2)
    # preprocessing
    X_train, X_test, Y_train, Y_test = preprocessing(X_train_org, X_test_org, Y_train_org, Y_test_org)

    model(X_train, Y_train[:, 6:10], X_test, Y_test[:, 6:10], epoches=2000)
