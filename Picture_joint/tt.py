# coding:utf-8
'''
Created on 2017/11/24

@author: sunyihuan
'''
from PIL import Image
import numpy as np
from skimage import transform, data
import scipy.io as scio
import tensorflow as tf

#
# target = Image.new('RGBA', (1000, 2000), (0, 0, 0, 0))
# target.show()
# image = Image.open('cartoon/face/face_1.png')
# image=target.paste(image, [0, 0], mask=image)
# image.show()
a = [1, 2]
b = [2, 3]
# print(np.min(np.array(a) / np.array(b)))

# print(np.array(a)*3)

face_norm = {
    'face': {'center': [1, 2], 'size': [30, 30]},
    'lip': {'center': [1, 2], 'size': [30, 30]},
    'nose': {'center': [1, 2], 'size': [30, 30]},
    'eyebrow': {'center': [1, 2], 'size': [30, 30]},
}

face_orign = {
    'face': {'center': [2, 2], 'size': [40, 30]},
    'lip': {'center': [1, 2], 'size': [40, 30]},
    'nose': {'center': [1, 2], 'size': [40, 30]},
    'eyebrow': {'center': [1, 2], 'size': [40, 30]},
}
org = 'lip'

# org_box = {}
#
# org_p = np.min(np.array(face_norm[org]['size']) / np.array(face_orign[org]['size']))
# #         # 移动
# org_box[org] = (np.array(face_norm[org]['size']) * org_p).astype(np.int32)
# print(org_box)

a = np.array([5, 4, 5, 3, 1, 5, 5, 5, 5, 7, 4, 4, 2, 4, 8, 5, 4, 4, 4, 5, 4, 2, 4, 5, 2, 4, 5, 8])
# a.astype(int)
# print(a)
# print(a.sum())
# file = 'E:/deeplearning_Data/face_1_channel_sense_XY64'
# data_train = scio.loadmat(file)
# X = data_train['X']
# Y = data_train['Y'].T
# res = list(np.argmax(Y.T, 1))
# num = len(res)
# classes = Y.shape[0]
# for i in range(classes):
#     print(str(i) + '的比例', round(100.0 * res.count(i) / num, 2), '%')

Y = scio.loadmat('E:/deeplearning_Data/face_1_channel_XY')
Y = Y['Y'][1200:, :]
print(tf.argmax(Y, 1))
