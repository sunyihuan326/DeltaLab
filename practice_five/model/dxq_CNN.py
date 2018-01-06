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


def model(trX, trY, lr=.5, epoches=200, minibatch_size=64):
    m, features = trY.shape
    X = tf.placeholder(tf.float32, shape=[None, 64 * 64 * 3])
    XX = tf.reshape(X, shape=[-1, 64, 64, 3])
    Y = tf.placeholder(tf.float32, shape=[None, features])
    # YY = tf.one_hot(Y, 10, on_value=1, off_value=None, axis=1)

    # dp = tf.placeholder(tf.float32)
    global_step = tf.Variable(0, trainable=False)

    reg1 = tf.contrib.layers.l2_regularizer(scale=0.1)
    conv1 = tf.layers.conv2d(XX, 32, 5, padding='same', activation=tf.nn.relu, kernel_regularizer=reg1)
    conv1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same')

    # conv2 = tf.layers.conv2d(conv1, 64, 3, padding='same', activation=tf.nn.relu)
    # conv2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same')

    # conv3 = tf.layers.conv2d(conv2, 128, 3, padding='same', activation=tf.nn.relu)
    # conv3 = tf.layers.average_pooling2d(conv3, 2, 2, padding='same')

    # convZ = tf.layers.flatten(pool3)
    convZ = tf.contrib.layers.flatten(conv1)

    fc1 = tf.layers.dense(convZ, 256, activation=tf.nn.relu, kernel_regularizer=reg1)
    # fc1 = tf.layers.batch_normalization(fc1)
    # fc1 = tf.layers.dropout(fc1, rate=dp, training=True)

    # fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu, kernel_regularizer=reg1)
    # fc2 = tf.layers.batch_normalization(fc2)
    # fc2 = tf.layers.dropout(fc2, rate=dp, training=True)
    #
    # fc3 = tf.layers.dense(fc2, 64, activation=tf.nn.relu, name='fc3')
    #
    # fc3 = tf.layers.batch_normalization(fc3)
    # fc3 = tf.layers.dropout(fc3, rate=dp, training=True)
    ZL = tf.layers.dense(fc1, 18, activation=None, name='output')
    print(ZL)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ZL, labels=Y))

    learning_rate = tf.train.exponential_decay(lr,
                                               global_step=global_step,
                                               decay_steps=100, decay_rate=0.9)
    learning_rate = tf.maximum(learning_rate, .001)
    loss = tf.reduce_mean(tf.square(Y - ZL))

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
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
                print('epoch', epoch, 'loss', minibatch_cost)
                print(llr)

        saver.save(sess, "save/model.ckpt")


if __name__ == '__main__':
    file = '../data/face_top_9.mat'
    data = scio.loadmat(file)
    trX = data['X'] / 255.
    trY = data['Y']
    model(trX, trY, epoches=1000)
