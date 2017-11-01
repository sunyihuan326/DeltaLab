# coding:utf-8
'''
Created on 2017/10/31.

@author: chk01
'''
import numpy as np


def normalize(x):
    # axis=1 行
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm

    return x


def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


def L1(yhat, y):
    loss = np.sum(np.abs(y - yhat))
    return loss


def L2(yhat, y):
    loss = np.dot((y - yhat), (y - yhat).T)
    return loss


if __name__ == '__main__':
    x = np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0, 0]])
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print(softmax(x))
    print("L1 = " + str(L1(yhat, y)))
    print("L2 = " + str(L2(yhat, y)))
