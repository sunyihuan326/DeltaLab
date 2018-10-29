# coding:utf-8
'''
Created on 2017/11/17

@author: sunyihuan
'''
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.spatial.distance import pdist, squareform

image1 = Image.open('syh_02.jpg').convert("L").resize([300, 300])
image2 = Image.open('syh.jpg').convert("L").resize([300, 300])
# HOG特征提取
im1 = hog(image1)
im2 = hog(image2)

# 计算曼哈顿距离
dis_mahadun = np.sum(np.abs(im1 - im2))
print(dis_mahadun)
# im = np.vstack([im1, im2])
# dis = pdist(im,'cityblock')


radius = 2
n_points = 8 * radius

# 计算欧式距离
dist = np.linalg.norm(im1 - im2)
print(dist)

# img = Image.open('lhf01.jpg')
# img_value = 255.*(np.array(img.convert("L"))>=60)
# Image.fromarray(img_value).show()
# # imlbp = local_binary_pattern(image, n_points, radius)

# plt.subplot(133)
# plt.imshow(imlbp, cmap='gray')
# plt.show()

# print(im1.shape)

