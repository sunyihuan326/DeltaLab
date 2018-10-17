# coding:utf-8 
'''
created on 2018/10/16

@author:sunyihuan
'''
import os
from tensorflow.python import pywrap_tensorflow

model_dir = "/Users/sunyihuan/Desktop/liuhai_model"
checkpoint_path = os.path.join(model_dir, "model.ckpt-37")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)

