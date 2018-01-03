# coding:utf-8
'''
Created on 2018/1/3.

@author: chk01
'''
import numpy as np


def mashi_distance(x, y):
    # 马氏距离要求样本数要大于维数，否则无法求协方差矩阵
    # 此处进行转置，表示10个样本，每个样本2维
    X = np.vstack([x, y])
    XT = X.T

    # 方法一：根据公式求解
    S = np.cov(X)  # 两个维度之间协方差矩阵
    SI = np.linalg.inv(S)  # 协方差矩阵的逆矩阵
    # 马氏距离计算两个样本之间的距离，此处共有4个样本，两两组合，共有6个距离。
    n = XT.shape[0]
    d1 = []
    for i in range(0, n):
        for j in range(i + 1, n):
            delta = XT[i] - XT[j]
            d = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
            d1.append(d)
    print(d1)


if __name__ == '__main__':
    a = np.array([[[1, 2], [2, 3]], [[2, 4], [1, 2]]])
    b = np.array([[2, 3], [1, 2]])
    print(a.shape)
    print(a - b)
    distinces = np.sum(np.linalg.norm(a - b, axis=1), axis=1)
    print('d', distinces)

    print(np.linalg.norm(a - b, axis=(1, 2)))
    # 第一列
    x = [3, 5, 2, 8]

    # 第二列
    y = [4, 6, 2, 4]

    # mashi_distance(x, y)
