# coding:utf-8
'''
Created on 2018/1/23.

@author: chk01
'''
import scipy.io as scio

pre = scio.loadmat('backup/chin_C.mat')
now = scio.loadmat('material/feature_matrix/chin_C.mat')
print(pre['X'][0]-now['X'][0])
print('==---------====')
print(now['X'][1])
