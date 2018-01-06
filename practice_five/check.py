# coding:utf-8
'''
Created on 2018/1/6.

@author: chk01
'''
import scipy.io as scio
import matplotlib.pyplot as plt

file = 'data/face_top_9.mat'
data = scio.loadmat(file)
points = data['Y']
for point in points:
    plt.scatter(point.reshape(-1, 2)[:, 0], -point.reshape(-1, 2)[:, 1])
    plt.show()
