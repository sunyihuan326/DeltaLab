# coding:utf-8
'''
Created on 2017/11/3

@author: sunyihuan
'''

import numpy as np

x = np.array([[1, 2, 3], [2, 3, 4]])
x0 = x.reshape(x.shape[0] * x.shape[1], 1)
print(x0,x0.shape)

print("w我是分割线")

x1 = x.reshape(x.shape[0] * x.shape[1], -1)
print(x1)
print(x1.shape)