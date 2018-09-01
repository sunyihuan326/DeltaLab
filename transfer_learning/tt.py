# coding:utf-8 
'''
created on 2018/8/31

@author:sunyihuan
'''
import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

# hair_length_graph = tf.Graph()
#
# # model_path = "/Users/sunyihuan/Desktop/parameters/hair_length/second/complex75.03_simple78.63/model.ckpt-23"  # check4
model_path = "/Users/sunyihuan/Desktop/parameters/hair_length/second/complex77_simple70/model.ckpt-39"  # check0
#
# with hair_length_graph.as_default():
#     saver = tf.train.import_meta_graph("{}.meta".format(model_path))
#     sess = tf.InteractiveSession(graph=hair_length_graph)
#     saver.restore(sess, model_path)

# var_to_shape_map = sess.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
#     print(sess.get_tensor(key))

from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(file_name=model_path, tensor_name='', all_tensors=False, all_tensor_names=True)
