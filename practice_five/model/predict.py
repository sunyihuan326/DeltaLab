# coding:utf-8
'''
Created on 2018/1/6.

@author: chk01
'''
import tensorflow as tf
import scipy.io as scio
import matplotlib.pyplot as plt


def main():
    file = '../data/face_top_9.mat'
    data = scio.loadmat(file)
    tf.reset_default_graph()
    # graph
    saver = tf.train.import_meta_graph("save/model-fc-250-10.ckpt.meta")
    # value
    # a = tf.train.NewCheckpointReader('save/model.ckpt.index')
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "save/model-fc-250-10.ckpt")
        graph = tf.get_default_graph()

        predict_op = graph.get_tensor_by_name("output/BiasAdd:0")
        # w = graph.get_collection('trainable_variables')[0]
        # b = graph.get_collection('trainable_variables')[1]
        # ww = graph.get_tensor_by_name("output/kernel:0")
        # print(w, b)
        # print(ww)
        X = graph.get_tensor_by_name("Placeholder:0")
        # dp = graph.get_tensor_by_name("Placeholder_2:0")
        # print(w.eval())
        # print(ww.eval())
        # print(b.eval())
        resY = predict_op.eval({X: data['X'][:50].reshape(-1, 64, 64, 3) / 255.})
        for i in range(10):
            plt.figure(i + 1)
            plt.scatter(resY[i].reshape(-1, 2)[:, 0], -resY[i].reshape(-1, 2)[:, 1])
            plt.scatter(data['Y'][i].reshape(-1, 2)[:, 0], -data['Y'][i].reshape(-1, 2)[:, 1])
            plt.savefig('test/' + str(i) + '.png')


if __name__ == '__main__':
    main()
