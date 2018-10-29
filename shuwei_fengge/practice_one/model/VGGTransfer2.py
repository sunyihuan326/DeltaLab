# coding:utf-8
'''
Created on 2017/12/7.

@author: chk01
'''
import tensorflow as tf
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import os
import math


class CONFIG:
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64
    COLOR_CHANNELS = 3


def load_vgg_model(path):
    vgg = scio.loadmat(path)

    vgg_layers = vgg['layers']

    def _weights(layer, expected_layer_name):
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(conv2d_layer):
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input'] = tf.placeholder(
        dtype='float32', shape=(None, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS))
    graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1'] = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])

    return graph, vgg_layers


def compute_cost(ZL, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(ZL), labels=tf.transpose(Y)))
    return cost


def init_sets(X, Y, file, distribute):
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    assert len(distribute) == 2
    assert sum(distribute) == 1
    scio.savemat(file + 'VGG_train',
                 {'X': shuffled_X[:, :int(m * distribute[0])], 'Y': shuffled_Y[:, :int(m * distribute[0])]})
    scio.savemat(file + 'VGG_test',
                 {'X': shuffled_X[:, int(m * distribute[0]):], 'Y': shuffled_Y[:, int(m * distribute[0]):]})
    return True


def load_data(file):
    if not os.path.exists(file + 'VGG_test.mat'):
        data = scio.loadmat(file)
        m = data['X'].shape[0]
        x = data['X'].reshape(m, -1).T
        y = np.squeeze(data['Y']).T
        init_sets(x, y, file, distribute=[0.8, 0.2])
    return True


def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


if __name__ == '__main__':
    name = 'Dxq'
    if name == 'Dxq':
        file = 'F:/dataSets/FaceChannel3/face_3_channel_XY64'
    is_training = True
    dropout = 0.1
    n_classes = 9
    initial_learning_rate = 0.5
    load_data(file)

    data_train = scio.loadmat(file + 'VGG_train')
    X_train = data_train['X'] / 255.
    Y_train = data_train['Y']

    data_test = scio.loadmat(file + 'VGG_test')
    X_test = data_test['X'] / 255.
    Y_test = data_test['Y']

    minibatch_size = 20
    minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

    # model_dir = "F:\AImodel\imagenet-vgg-verydeep-19.mat"
    # model, vgg_layers = load_vgg_model(model_dir)
    # tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    Y = tf.placeholder(tf.float32, shape=[None, n_classes])

    global_step = tf.Variable(0, trainable=False)

    # vgg_out = model['input']
    vgg_out = X
    _, k1, k2, channel = vgg_out.shape
    #  (8, 8, 8, 512)
    fc1 = tf.contrib.layers.flatten(vgg_out)
    # Z1 = tf.layers.dense(s2, 128, activation=tf.nn.relu)
    # fc1 = tf.layers.dense(fc1, 128)
    # fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
    out = tf.layers.dense(fc1, n_classes, activation=None)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(out), labels=tf.transpose(Y)))
    tf.summary.scalar('cost', cost)

    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10, decay_rate=0.98)

    optimizer = tf.train.AdamOptimizer(learning_rate + 0.001)
    train_step = optimizer.minimize(cost)

    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    add_global = global_step.assign_add(1)

    init = tf.global_variables_initializer()
    merge_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(init)
        Writer = tf.summary.FileWriter(logdir='logdir/VGG', graph=sess.graph)
        for epoch in range(20):
            minibatch_cost = 0
            num_minibatches = int(1500 / minibatch_size)
            for i, minibatch in enumerate(minibatches):
                (minibatch_X, minibatch_Y) = minibatch
                summary, cs, _, global_step = sess.run([merge_op, cost, train_step, add_global],
                                                       feed_dict={X: minibatch_X.T.reshape(-1, 64, 64, 3),
                                                                  Y: minibatch_Y.T})
                Writer.add_summary(summary, global_step)

                # o = res.eval(feed_dict={X: minibatch_X.T.reshape(-1, 64, 64, 3),
                #                         Y: minibatch_Y.T})
                # t = resY.eval(feed_dict={X: minibatch_X.T.reshape(-1, 64, 64, 3),
                #                          Y: minibatch_Y.T})
                #
                # minibatch_cost += cs / num_minibatches
            a = accuracy.eval(feed_dict={X: X_train.T.reshape(-1, 64, 64, 3), Y: Y_train.T})
            if epoch % 1 == 0:
                print("Cost after epoch %i: %f" % (epoch, a))
