# coding:utf-8 
'''
created on 2018/3/9

@author:Dxq
'''
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

data = scio.loadmat('F:/dataSets/FaceChannel3/face_3_channel_XY64.mat')
print(data['X'].shape)
print(data['Y'].shape)
# print(np.mod(0, 3))

Z = np.argmax(data['Y'], 1)
ZZ = np.zeros([1502, 1])
for i in range(len(Z)):
    ZZ[i] = [0, 1, 2][int(np.mod(Z[i], 3))]
    # 151,658,693
for i in range(9):
    print(len(Z[np.flatnonzero(Z == i)]))
# assert 1 == 2
# scio.savemat('F:/dataSets/FaceChannel3/64X64X3-XY-Sense.mat', {"X": data['X'], "Y": ZZ})
# Image.fromarray(np.uint8(data['X'][1].reshape(64, 64, 3)), mode='RGB').show()
