# coding:utf-8
'''
Created on 2017/11/8

@author: sunyihuan
'''
import numpy as np
A=np.array([[1,2,3],[2,3,5]])
# print(A.shape)
B=np.random.rand(A.shape[0],A.shape[1])
# print(B)

A=(A<0.5)
print(A)