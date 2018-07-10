# coding:utf-8 
'''
created on 2018/7/10

@author:sunyihuan

do not finish
'''

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope

from tensorflow.contrib.layers import conv2d, max_pool2d


@add_arg_scope
def inception_layer(inputs,
                    conv_11_size,
                    conv_33_reduce_size, conv_33_size,
                    conv_55_reduce_size, conv_55_size,
                    pool_size,
                    data_dict={},
                    trainable=False,
                    name='inception'):
    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([conv2d], nl=tf.nn.relu, trainable=trainable,
                   data_dict=data_dict):
        conv_11 = conv2d(inputs, 1, conv_11_size, '{}_1x1'.format(name))

        conv_33_reduce = conv2d(inputs, 1, conv_33_reduce_size,
                                '{}_3x3_reduce'.format(name))
        conv_33 = conv2d(conv_33_reduce, 3, conv_33_size, '{}_3x3'.format(name))

        conv_55_reduce = conv2d(inputs, 1, conv_55_reduce_size,
                                '{}_5x5_reduce'.format(name))
        conv_55 = conv2d(conv_55_reduce, 5, conv_55_size, '{}_5x5'.format(name))

        pool = max_pool2d(inputs, '{}_pool'.format(name), stride=1,
                          padding='SAME', filter_size=3)
        convpool = conv2d(pool, 1, pool_size, '{}_pool_proj'.format(name))

    return tf.concat([conv_11, conv_33, conv_55, convpool],
                     3, name='{}_concat'.format(name))


MEAN = [103.939, 116.779, 123.68]


def resize_tensor_image_with_smallest_side(image, small_size):
    """
    Resize image tensor with smallest side = small_size and
    keep the original aspect ratio.

    Args:
        image (tf.tensor): 4-D Tensor of shape
            [batch, height, width, channels] or 3-D Tensor of shape
            [height, width, channels].
        small_size (int): A 1-D int. The smallest side of resize image.

    Returns:
        Image tensor with the original aspect ratio and
        smallest side = small_size .
        If images was 4-D, a 4-D float Tensor of shape
        [batch, new_height, new_width, channels].
        If images was 3-D, a 3-D float Tensor of shape
        [new_height, new_width, channels].
    """
    im_shape = tf.shape(image)
    shape_dim = image.get_shape()
    if len(shape_dim) <= 3:
        height = tf.cast(im_shape[0], tf.float32)
        width = tf.cast(im_shape[1], tf.float32)
    else:
        height = tf.cast(im_shape[1], tf.float32)
        width = tf.cast(im_shape[2], tf.float32)

    height_smaller_than_width = tf.less_equal(height, width)

    new_shorter_edge = tf.constant(small_size, tf.float32)
    new_height, new_width = tf.cond(
        height_smaller_than_width,
        lambda: (new_shorter_edge, (width / height) * new_shorter_edge),
        lambda: ((height / width) * new_shorter_edge, new_shorter_edge))

    return tf.image.resize_images(
        tf.cast(image, tf.float32),
        [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)])
