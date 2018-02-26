# coding:utf-8 
'''
created on 2018/2/24

@author:sunyihuan
'''
import tensorflow as tf
import os
from tensorflow.python.tools import inspect_checkpoint as chkp

logdir = '/Users/sunyihuan/Desktop/孙义环/mnist-model/mnist_data'
# chkp.print_tensors_in_checkpoint_file(logdir+'/model_epoch_0008_step_3860',tensor_name='PrimaryCaps_layer/Conv/weights',all_tensors=False)
tt=os.listdir(logdir)
with tf.Session() as sess:
    # saver = tf.train.Saver()
    print(tf.train.latest_checkpoint(logdir))
    saver = tf.train.import_meta_graph(logdir+'/model_epoch_0008_step_3860.meta')

    saver.restore(sess,logdir+'/model_epoch_0008_step_3860')
    graph = tf.get_default_graph()
    print(graph.get_operations())
    for g in graph.get_operations():
        print(g.name)
    print(tf.import_graph_def())
