# coding:utf-8
'''
Created on 2017/11/24

@author: sunyihuan
'''
from PIL import Image
import numpy as np
from skimage import transform, data

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
org='lip'

org_box = {}

org_p = np.min(np.array(face_norm[org]['size']) / np.array(face_orign[org]['size']))
#         # 移动
org_box[org] = (np.array(face_norm[org]['size']) * org_p).astype(np.int32)
print(org_box)