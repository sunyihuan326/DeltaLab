# coding:utf-8
'''
Created on 2017/12/8.

@author: chk01
'''
data = []
import numpy as np


def main():
    return np.random.random()

while len(data) < 10:
    res = main()
    if res > 0.5:
        data.append(res)
print(data)
