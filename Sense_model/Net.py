# coding:utf-8 
'''
created on 2018/3/8

@author:sunyihuan
'''
import tensorflow as tf
import Sense_model.config as config
from Sense_model.utils import load_data, preprocessing

cfg = config.cfg
epsilon = 1e-9

file = "/Users/sunyihuan/PycharmProjects/DeltaLab/Sense_model/data/face_1_channel_XY64_sense.mat"


class Net(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                # self.X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
                # self.labels = tf.placeholder(tf.int64, shape=(None,))
                self.X, self.labels = preprocessing(load_data(file))
                self.Y = tf.one_hot(self.labels, depth=3, axis=1, dtype=tf.float32)
                self.build_arch()
                self.loss()
                self._summary()

                # t_vars = tf.trainable_variables()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss,
                                                        global_step=self.global_step)  # var_list=t_vars)
                # self.step_op = self.global_step.assign_add(1)
                # self.init_op = tf.global_variables_initializer()
            else:
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 28, 28, 1))
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
                self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, 10, 1))
                self.build_arch()

        tf.logging.info('Setting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')

        with tf.variable_scope('FC'):
            flaten = tf.layers.flatten(conv1)
            self.fc1 = tf.layers.dense(flaten, 20, activation=tf.nn.relu)
            self.out = tf.layers.dense(self.fc1, 10)

    def loss(self):
        self.total_loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.out)

    # Summary
    def _summary(self):
        pass
        train_summary = []
        # train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        # train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        # recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 28, 28, 1))
        # train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)
        #
        self.argmax_idx = tf.to_int32(tf.argmax(self.out, axis=1))
        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc')
