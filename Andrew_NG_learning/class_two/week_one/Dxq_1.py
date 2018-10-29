# coding:utf-8
'''
Created on 2017/11/4.

@author: chk01
'''
import numpy as np

x = np.array([1, 2, 3, 0, np.nan])
print(np.nan)
y1 = np.sum(x)
y2 = np.nansum(x)
print(y1)
print(y2)
print(range(2))
for i in range(2):
    print(i)