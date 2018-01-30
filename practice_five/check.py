# coding:utf-8
'''
Created on 2018/1/6.

@author: chk01
'''
import scipy.io as scio
import matplotlib.pyplot as plt
from PIL import Image

file = 'data/face_top_9.mat'
data = scio.loadmat(file)
points = data['Y']
imgs = data['X'].reshape(-1, 64, 64, 3)
r, g, b = imgs[:, :, :, 0], imgs[:, :, :, 1], imgs[:, :, :, 2]
img_L = 0.299 * r + 0.587 * g + 0.114 * b
print(img_L.shape)
print(points.shape)
scio.savemat('data/face_top_9_L.mat', {"X": img_L.reshape(-1, 64 * 64), "Y": points})
# plt.imshow(img_L[4], cmap='gray')
# plt.show()
# for point in points:
#     plt.scatter(point.reshape(-1, 2)[:, 0], -point.reshape(-1, 2)[:, 1])
#     plt.show()
