# coding:utf-8 
'''
created on 2018/3/28

@author:sunyihuan
'''
import tensorflow as tf
import numpy as np
import scipy.io as scio
import time
import random
import string
from PIL import Image
import os
import base64
from urllib.parse import urlencode
from Tencent.utils import *

# def get_params(plus_item):
#     '''''请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效）'''
#     t = time.time()
#     time_stamp = str(int(t))
#
#     '''''请求随机字符串，用于保证签名不可预测'''
#     nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 10))
#
#     '''''应用标志，这里修改成自己的id和key'''
#     app_id = '1256126892'
#     uid = "100004156358"
#     project_id = "project_id"
#     model_id = "model_id"
#     rsp_img_type = "url"
#     app_key = 'qWkUndEe5DTumeQW'
#
#     params = {'app_id': app_id,
#               'time_stamp': time_stamp,
#               'nonce_str': nonce_str,
#               'image': get_img_base64str(plus_item),
#               'mode': 0
#               }
#     sign = gen_dict_md5(params, app_key)
#     params['sign'] = sign
#
#     return params

x1 = np.array([[1, 2], [3, 4]])
x2 = np.array([[2, 3], [3, 5]])

b = tf.concat([x1, x2], axis=0)
c = tf.concat([x1, x2], 1)
train = scio.loadmat("/Users/sunyihuan/Desktop/Data/qx_top/qx_top64/64-train.mat")
x_train = train["X"][:209]
test = scio.loadmat("/Users/sunyihuan/Desktop/Data/qx_top/qx_top64/64-test.mat")
x_test = test["X"]
print(x_train.shape, x_test.shape)

b_x = tf.concat([x_train, x_test], axis=3)
with tf.Session() as sess:
    b0 = sess.run(b)
    c0 = sess.run(c)
    bx = sess.run(b_x)
# print(b0.shape)
# print(c0.shape)
print(bx.shape)
