# coding:utf-8 
'''
created on 2018/3/8

@author:sunyihuan
'''

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string(flag_name='results', default_value='results', docstring='all of results will be saved to directory')
flags.DEFINE_string('dataset', 'mnist', 'The name of dataset')
flags.DEFINE_integer(flag_name='batch_size', default_value=128, docstring='')
flags.DEFINE_integer('num_threads', 1, 'number of threads of enqueueing examples')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_boolean('is_training', False, 'train or predict phase')
flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 100, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 1, 'the frequency of saving model(epoch)')
flags.DEFINE_integer(flag_name='epoch', default_value=2, docstring='')
# flags.DEFINE_string(flag_name='mnist_train_record',
#                     default_value='C:/Users/chk01/Desktop/mnist_data/train.tfrecords',
#                     docstring='')
# flags.DEFINE_string(flag_name='mnist_test_record',
#                     default_value='C:/Users/chk01/Desktop/mnist_data/test.tfrecords',
#                     docstring='')
# flags.DEFINE_string(flag_name='mnist_valid_record',
#                     default_value='C:/Users/chk01/Desktop/mnist_data/valid.tfrecords',
#                     docstring='')

cfg = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
