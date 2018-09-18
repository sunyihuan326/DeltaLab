# coding:utf-8 
'''
created on 2018/8/31

@author:sunyihuan
'''
import os
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np

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

# 查看ckpt中的tensor
# from tensorflow.python.tools import inspect_checkpoint as chkp

# chkp.print_tensors_in_checkpoint_file(file_name=model_path, tensor_name='', all_tensors=False, all_tensor_names=True)

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
total = a + b
# print(a)
# print(b)
# print(total)
sess = tf.Session()
# print(sess.run(total))
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
print(linear_model)
y = linear_model(x)
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6],[7,8,9]]}))

features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))
print(sess.run(inputs))


