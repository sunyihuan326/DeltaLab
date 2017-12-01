# coding:utf-8
'''
Created on 2017/11/15.

@author: chk01
'''
import scipy.io as scio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
pathDir = 'F:/dataSets/wiki/'
# data = scio.loadmat('wiki.mat')
# img = np.squeeze(data['wiki'][0][0][2])
# gender = np.squeeze(data['wiki'][0][0][3])
# img_data = []
# gender_data = []
# for i in range(img.shape[0]):
#     img_data.append(img[i][0])
# for j in range(gender.shape[0]):
#     gender_data.append(gender[j])
# scio.savemat('gender.mat', {'Img': np.array(img_data), 'Gender': gender_data})
# data = scio.loadmat('gender.mat')
# print(Image.open(pathDir + data['Img'][0]))
# print(data['Gender'][0][0])


