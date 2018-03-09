# coding:utf-8 
'''
created on 2018/2/24

@author:Dxq
'''
import tensorflow as tf
from NetTest.config import cfg


def load_data(dataset='eyelid', train_mode='train'):
    if dataset == 'eyelid':
        if train_mode == 'train':
            record_file = cfg.eyelid_train_record
        elif train_mode == 'valid':
            record_file = cfg.eyelid_valid_record
        else:
            record_file = cfg.eyelid_test_record
    elif dataset == 'mnist':
        if train_mode == 'train':
            record_file = cfg.mnist_train_record
        elif train_mode == 'valid':
            record_file = cfg.mnist_valid_record
        else:
            record_file = cfg.mnist_test_record
    file_queue = tf.train.string_input_producer([record_file], shuffle=True, capacity=500)
    reader = tf.TFRecordReader()
    key, _serialized = reader.read(file_queue)
    features = tf.parse_single_example(
        _serialized, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    # image = tf.decode_raw(features['image_raw'], tf.uint8)

    label = tf.cast(features['label'], tf.int64)
    image_dim = 28
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.cast(image, tf.float32) * (1. / 255)

    image = tf.reshape(image, [image_dim, image_dim, 1])
    image.set_shape([image_dim, image_dim, 1])

    return image, label


def get_batch_data(dataset='eyelid', batch_size=128, num_threads=3, train_mode='train', graph=tf.get_default_graph()):
    with graph.as_default():
        image, label = load_data(dataset, train_mode)

        if train_mode in ['train', 'valid']:
            X, Y = tf.train.shuffle_batch([image, label], num_threads=num_threads,
                                          batch_size=batch_size,
                                          capacity=batch_size * 32 + num_threads * batch_size,
                                          allow_smaller_final_batch=False,
                                          min_after_dequeue=batch_size * 32)
        else:
            X, Y = tf.train.batch([image, label], num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64)
    return X, Y


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)
