# coding:utf-8
'''
Created on 2017/12/19

@author: sunyihuan
'''
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE

logdir = 'save'


def preprocessing(trX, teX, trY, teY):
    m, _ = trX.shape
    # res = RandomOverSampler(ratio={0: 4*int(0.5*m), 1: 4*int(0.3*m), 2: 4*int(0.2*m)})
    # trX, trY = res.fit_sample(trX, np.squeeze(trY))

    trX = trX / 255.
    teX = teX / 255.

    return trX, teX, trY, teY


def data_check(data, classes):
    num = len(data)

    for i in range(classes):
        print(str(i) + '的比例', round(100.0 * len(np.flatnonzero(data == i)) / num, 2), '%')
    print('<------------------分割线---------------------->')


# show data
# X_data = np.reshape(X_data, (-1, 28, 28))
def show_data(X, Y):
    for i in range(1, 10):
        plt.subplot(330 + i)
        plt.imshow(X[i], cmap=plt.get_cmap('gray'))
        plt.title(Y[i])
    plt.show()


def one_hot(y, classes):
    # m, _ = y.reshape(-1, 1).shape
    return np.eye(classes)[y]


def minibatches(X, Y, batch_size=64, shuffle=True):
    assert len(X) == len(Y)
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
    for start_idx in range(0, len(X) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield X[excerpt], Y[excerpt]


def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.div(tf.nn.l2_loss(features - centers_batch), int(len_features))
    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features
    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)
    return loss, centers_update_op


def model(trX, trY, teX, teY, lr=0.2, epoches=200, minibatch_size=128, drop_prob=.2):
    graph = tf.Graph()
    with graph.as_default():
        X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        # XX = tf.reshape(X, shape=[-1, 64, 64, 3])
        Y = tf.placeholder(tf.int32, shape=[None, 1])
        YY = tf.reshape(tf.one_hot(Y, 3, on_value=1, off_value=None, axis=1), [-1, 3])

        dp = tf.placeholder(tf.float32)
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

        reg_conv1 = tf.contrib.layers.l2_regularizer(scale=0.25)
        conv1 = tf.layers.conv2d(X, 32, 3, padding='same', kernel_regularizer=reg_conv1, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same')
        conv2 = tf.layers.conv2d(conv2, 64, 3, padding='same', kernel_regularizer=reg_conv1, activation=tf.nn.relu)

        # conv1 = tf.layers.dropout(conv1, rate=dp, training=True)
        # conv1 = tf.layers.batch_normalization(conv1)

        reg2 = tf.contrib.layers.l2_regularizer(scale=0.2)
        conv3 = tf.layers.conv2d(conv2, 64, 3, padding='same', kernel_regularizer=reg2, activation=tf.nn.relu)
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2, padding='same')
        # conv3 = tf.layers.batch_normalization(conv3)

        # reg3 = tf.contrib.layers.l2_regularizer(scale=0.2)
        # conv3 = tf.layers.conv2d(conv2, 36, 5, padding='same', kernel_regularizer=reg3, activation=tf.nn.relu)
        # conv3 = tf.layers.max_pooling2d(conv3, 2, 2, padding='same')

        # convZ = tf.layers.flatten(pool3)
        convZ = tf.contrib.layers.flatten(conv3)

        fc1 = tf.layers.dense(convZ, 16, activation=tf.nn.relu)
        # fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)
        # fc3 = tf.layers.dense(fc2, 64, activation=tf.nn.relu)
        # fc4 = tf.layers.dense(fc3, 32, activation=tf.nn.relu)
        # fc1 = tf.layers.batch_normalization(fc1)
        fc1 = tf.layers.dropout(fc1, rate=dp, training=True)
        #
        # reg_fc2 = tf.contrib.layers.l2_regularizer(scale=0.8)
        # fc2 = tf.layers.dense(convZ, 128, kernel_regularizer=reg_fc2, activation=tf.nn.relu)
        # fc2 = tf.layers.batch_normalization(fc2)

        # reg_fc3 = tf.contrib.layers.l2_regularizer(scale=0.5)
        # fc3 = tf.layers.dense(fc2, 64, kernel_regularizer=reg_fc3)
        # fc2 = tf.layers.dropout(fc2, rate=dp, training=True)

        # fc5 = tf.layers.dense(convZ, 16, activation=None, name='fc3')

        # fc3_out = tf.nn.relu(fc5)
        # fc3 = tf.layers.batch_normalization(fc3)
        # fc3 = tf.layers.dropout(fc3, rate=dp, training=True)
        ZL = tf.layers.dense(fc1, 3, activation=None)

        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ZL, labels=Y))

        learning_rate = tf.train.exponential_decay(lr,
                                                   global_step=global_step,
                                                   decay_steps=100, decay_rate=0.9)
        learning_rate = tf.maximum(learning_rate, .00567)

        # with tf.variable_scope('loss_scope'):
        #     centerloss, centers_update_op = get_center_loss(ZL, Y, 0.5, 2)
        # self.loss = tf.losses.softmax_cross_entropy(onehot_labels=util.makeonehot(self.y, self.CLASSNUM), logits=self.score)
        # lambda则0.1-0.0001之间不等
        # print('YY', YY)
        # print('ZL', ZL)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=ZL)
        with tf.control_dependencies([]):  # centers_update_op
            train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)

        #     train_op = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=ZL)
        # train_op = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss, global_step=global_step)

        predict_op = tf.argmax(ZL, 1)
        print('predict_op', predict_op)
        correct_prediction = tf.equal(predict_op, tf.argmax(YY, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('accuracy', accuracy)
        # add_global = global_step.assign_add(1)
        # init = tf.global_variables_initializer()
        sv = tf.train.Supervisor(graph=graph, logdir=logdir, save_model_secs=0)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            # sess.run(init)
            for epoch in range(epoches):
                for minibatch_X, minibatch_Y in minibatches(trX, trY, minibatch_size, shuffle=True):
                    _lr, _loss, _ = sess.run([learning_rate, loss, train_op],
                                             feed_dict={X: minibatch_X, Y: minibatch_Y, dp: drop_prob})
                if (epoch + 1) % 20 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={X: trX, Y: trY, dp: 0.0})
                    test_accuracy = sess.run(accuracy, feed_dict={X: teX, Y: teY, dp: 0.0})
                    print('lr===', _lr)
                    print(
                        "Cost after epoch %i: %f tr-acc: %f te-acc: %f" % (
                            epoch + 1, _loss, train_accuracy, test_accuracy))
                    # print("X", trX[2:5])
                    # print("conv1", sess.run(conv1, feed_dict={X: trX[2:5], Y: trY[2:5], dp: 0.0}))
                    # print("convZ", sess.run(convZ, feed_dict={X: trX[2:5], Y: trY[2:5], dp: 0.0}))
                    # print("fc2", sess.run(fc2, feed_dict={X: trX[2:5], Y: trY[2:5], dp: 0.0}))
                    # print("predict_op", predict_op.eval({X: trX[2:10]}))
                    # print(np.squeeze(trY[2:10]))
                    sv.saver.save(sess, logdir + "/model-{}-{}".format(epoch + 1, round(test_accuracy * 100, 2)))

            # 修改网络倒数层为2，然后输出特征
            # _fc3 = fc3.eval({X: teX[:2000], Y: teY[:2000], dp: 0.0})
            # plt.scatter(_fc3[:, 0], _fc3[:, 1], c=teY[:2000])
            # plt.show()
            train_accuracy = sess.run(accuracy, feed_dict={X: trX[:2000], Y: trY[:2000], dp: 0.0})
            test_accuracy = sess.run(accuracy, feed_dict={X: teX[:2000], Y: teY[:2000], dp: 0.0})
            print("Train Accuracy:", train_accuracy)
            print("Test Accuracy:", test_accuracy)
            # saver.save(sess, "save/model-{}-{}".format(epoches, round(test_accuracy, 2)))


def predict(_X, _Y):
    from PIL import Image
    tf.reset_default_graph()
    # graph
    modle_name = 'model-80-66.18'
    saver = tf.train.import_meta_graph("best/{}.meta".format(modle_name))
    # value
    # a = tf.train.NewCheckpointReader('save/model.ckpt.index')
    # saver = tf.train.Saver()
    _X = (_X / 255.).reshape(-1, 64, 64, 3)
    _Y = _Y.reshape(-1, 1)
    with tf.Session() as sess:
        saver.restore(sess, "best/" + modle_name)
        graph = tf.get_default_graph()
        predict_op = graph.get_tensor_by_name("ArgMax:0")
        X = graph.get_tensor_by_name("Placeholder:0")
        dp = graph.get_tensor_by_name("Placeholder_2:0")
        prediction = predict_op.eval({X: _X, dp: 0.0})
        print('valid_accuracy==', sum(prediction == np.squeeze(_Y)) / len(_X))

        for i in range(len(_X)):
            if prediction[i] != np.squeeze(_Y)[i]:
                # print(np.squeeze(vaLabel)[i])
                Image.fromarray(np.uint8(_X[i] * 255), 'RGB').save('error/pre{}-cor{}-{}.jpg'.format(prediction[i],
                                                                                                     int(np.squeeze(
                                                                                                         _Y)[i]),
                                                                                                     i))
        train_res_matrix = confusion_matrix(y_true=prediction, y_pred=_Y)
        print(train_res_matrix)


trD = scio.loadmat('F:/dataSets/FaceChannel3/64X64X3-XY-Sense-train.mat')
trX = trD['X']
trY = trD['Y']

vaD = scio.loadmat('F:/dataSets/FaceChannel3/64X64X3-XY-Sense-valid.mat')
vaX = vaD['X']
vaY = vaD['Y']

teD = scio.loadmat('F:/dataSets/FaceChannel3/64X64X3-XY-Sense-test.mat')
teX = teD['X']
teY = teD['Y']

data_check(trY, 3)
data_check(vaY, 3)
data_check(teY, 3)

# predict(teX, teY)
# assert 1 == 0
X_train, X_test, Y_train, Y_test = preprocessing(trX, teX, trY, teY)
model(X_train.reshape(-1, 64, 64, 3), Y_train.reshape(-1, 1), X_test.reshape(-1, 64, 64, 3), Y_test, epoches=5000,
      drop_prob=0.25)
