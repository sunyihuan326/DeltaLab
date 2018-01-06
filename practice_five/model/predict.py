# coding:utf-8
'''
Created on 2018/1/6.

@author: chk01
'''
import tensorflow as tf
import scipy.io as scio


def main():
    file = '../data/face_top_9.mat'
    data = scio.loadmat(file)
    tf.reset_default_graph()
    # graph
    saver = tf.train.import_meta_graph("save/model.ckpt.meta")
    # value
    # a = tf.train.NewCheckpointReader('save/model.ckpt.index')
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "save/model.ckpt")
        graph = tf.get_default_graph()

        predict_op = graph.get_tensor_by_name("output/BiasAdd:0")
        X = graph.get_tensor_by_name("Placeholder:0")
        # dp = graph.get_tensor_by_name("Placeholder_2:0")

        resY = predict_op.eval({X: data['X']/255.})
        print(resY[0])


if __name__ == '__main__':
    main()
