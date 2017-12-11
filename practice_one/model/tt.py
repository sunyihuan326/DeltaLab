# coding:utf-8
'''
Created on 2017/12/8.

@author: chk01
'''
import scipy.io as scio

# file = 'F:/dataSets/MNIST/mnist_data_small'
# data = scio.loadmat(file)
# from sklearn.model_selection import train_test_split
#
# print(data['X'].shape)
# print(data['Y'].shape)
# X_train, X_test, Y_train, Y_test = train_test_split(data['X'], data['Y'], test_size=0.2)
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)
import numpy as np
ss = np.array([[1, 1], [2, 1], [1, 1]])
print(ss)
tt=np.argmax(ss)
print(tt)
