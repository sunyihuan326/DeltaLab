# coding:utf-8 
'''
created on 2018/2/24

@author:Dxq
'''
import tensorflow as tf
from eyelid_model.model.config import cfg
from eyelid_model.model.utils import get_batch_data

epsilon = 1e-9


class DxqNet(object):
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                # self.X = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
                # self.labels = tf.placeholder(tf.int64, shape=(None,))
                self.X, self.labels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads, train_mode='train',
                                                     graph=self.graph)
                self.valX, self.vallabels = get_batch_data(cfg.dataset, cfg.batch_size, cfg.num_threads,
                                                           train_mode='test', graph=self.graph)
                self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)
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
                self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 64, 64, 1))
                self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size,))
                self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, 2, 1))
                self.build_arch()

        tf.logging.info('Setting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            conv1 = tf.layers.conv2d(self.X, 24, 5, padding='SAME', activation=tf.nn.relu)
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='SAME')

        with tf.variable_scope('Conv2_layer'):
            conv2 = tf.layers.conv2d(conv1, 36, 5, padding='SAME', activation=tf.nn.relu)
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='SAME')

        with tf.variable_scope('Conv3_layer'):
            conv3 = tf.layers.conv2d(conv2, 48, 5, padding='SAME', activation=tf.nn.relu)
            conv3 = tf.layers.max_pooling2d(conv3, 2, 2, padding='SAME')

        with tf.variable_scope('FC'):
            convZ = tf.layers.flatten(conv3)
            fc1 = tf.layers.dense(convZ, 4, activation=tf.nn.relu)
            self.fc1 = tf.layers.batch_normalization(fc1)
            self.out = tf.layers.dense(self.fc1, 2)

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
