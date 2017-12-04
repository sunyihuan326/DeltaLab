# coding:utf-8
'''
Created on 2017/12/4.

@author: chk01
'''
import scipy.io as scio
import os
import matplotlib.pyplot as plt

# logdir = '../data/image3channel'
# files = os.listdir(logdir)
# for file in files:
#     _, _, id = file.replace('.mat', '').split('-')
#     if int(id) == 100002:
#         points = scio.loadmat(logdir + '/' + file)['Points']
#         plt.scatter(points[:, 0], points[:, 1])
#         plt.show()
tt=scio.loadmat('face_3_channel.mat')
print(tt['X'].shape)