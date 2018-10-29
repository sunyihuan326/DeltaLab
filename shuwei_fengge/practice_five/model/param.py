# coding:utf-8
'''
Created on 2018/1/17.

@author: chk01
'''
import tensorflow as tf
import scipy.io as scio


def save_parm2_mat():
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("save/model-fc-500.ckpt.meta")

    with tf.Session() as sess:
        saver.restore(sess, "save/model-fc-500.ckpt")
        graph = tf.get_default_graph()
        # print(graph.get_collection('trainable_variables'))
        # w = graph.get_collection('trainable_variables')[0]
        # b = graph.get_collection('trainable_variables')[1]
        tensor_w = graph.get_tensor_by_name("output/kernel:0")
        tensor_b = graph.get_tensor_by_name("output/bias:0")
        w = tensor_w.eval()
        b = tensor_b.eval()
        scio.savemat('param', {"W": w, "b": b.reshape(1, -1)})


def save_vgg19_parm2_mat():
    # 0, 'conv1_1'
    # 2, 'conv1_2'
    # 5, 'conv2_1'
    # 7, 'conv2_2'
    # 10, 'conv3_1'
    # 12, 'conv3_2'
    # 14, 'conv3_3'
    # 16, 'conv3_4'
    # 19, 'conv4_1'
    # 21, 'conv4_2'
    # 23, 'conv4_3'
    # 25, 'conv4_4'
    # 28, 'conv5_1'
    # 30, 'conv5_2'
    # 32, 'conv5_3'
    # 34, 'conv5_4'
    vgg = scio.loadmat("F:/AImodel/imagenet-vgg-verydeep-19.mat")
    vgg_layers = vgg['layers']
    for layer in [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]:
        wb = vgg_layers[0][layer][0][0][2]
        W = wb[0][0]
        b = wb[0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        scio.savemat(str(layer) + '-layer-parm', {"W": W, "b": b})
        # assert layer_name == expected_layer_name


if __name__ == '__main__':
    save_vgg19_parm2_mat()
