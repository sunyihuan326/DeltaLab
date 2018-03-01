# coding:utf-8 
'''
created on 2018/2/24

@author:Dxq
'''
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string(flag_name='results', default_value='results', docstring='all of results will be saved to directory')
flags.DEFINE_string('dataset', 'eyelid', 'The name of dataset')
flags.DEFINE_integer(flag_name='batch_size', default_value=64, docstring='')
flags.DEFINE_integer('num_threads', 30, 'number of threads of enqueueing examples')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_boolean('is_training', False, 'train or predict phase')
flags.DEFINE_integer('train_sum_freq', 10, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 20, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 1, 'the frequency of saving model(epoch)')
flags.DEFINE_integer(flag_name='epoch', default_value=10, docstring='')
flags.DEFINE_string(flag_name='eyelid_train_record',
                    default_value='C:/Users/chk01/Desktop/eyelid/data/train.tfrecords',
                    docstring='')
flags.DEFINE_string(flag_name='eyelid_test_record',
                    default_value='C:/Users/chk01/Desktop/eyelid/data/valid.tfrecords',
                    docstring='')
flags.DEFINE_string(flag_name='eyelid_valid_record',
                    default_value='C:/Users/chk01/Desktop/eyelid/data/valid.tfrecords',
                    docstring='')

cfg = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)
