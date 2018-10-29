# coding:utf-8 
'''
created on 2018/6/21

@author:sunyihuan
'''
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist = read_data_sets("MNIST_data/", one_hot=True)


# 把数据写入Example
def get_tfrecords_example(feature, label):
    tfrecords_features = {}
    feat_shape = feature.shape
    tfrecords_features['feature'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.tostring()]))
    tfrecords_features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat_shape)))
    tfrecords_features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=label))
    return tf.train.Example(features=tf.train.Features(feature=tfrecords_features))


# 把所有数据写入tfrecord文件
def make_tfrecord(data, outf_nm='mnist-train'):
    feats, labels = data
    outf_nm += '.tfrecord'
    tfrecord_wrt = tf.python_io.TFRecordWriter(outf_nm)
    ndatas = len(labels)
    for inx in range(ndatas):
        exmp = get_tfrecords_example(feats[inx], labels[inx])
        exmp_serial = exmp.SerializeToString()
        tfrecord_wrt.write(exmp_serial)
    tfrecord_wrt.close()


import random

nDatas = len(mnist.train.labels)
inx_lst = range(nDatas)
random.shuffle(inx_lst)
random.shuffle(inx_lst)
ntrains = int(0.85 * nDatas)

# make training set
data = ([mnist.train.images[i] for i in inx_lst[:ntrains]], [mnist.train.labels[i] for i in inx_lst[:ntrains]])
make_tfrecord(data, outf_nm='mnist-train')

# make validation set
data = ([mnist.train.images[i] for i in inx_lst[ntrains:]], [mnist.train.labels[i] for i in inx_lst[ntrains:]])
make_tfrecord(data, outf_nm='mnist-val')

# make test set
data = (mnist.test.images, mnist.test.labels)
make_tfrecord(data, outf_nm='mnist-test')
