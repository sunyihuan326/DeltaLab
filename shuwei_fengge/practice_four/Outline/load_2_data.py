# coding:utf-8
'''
Created on 2018/1/6.

@author: chk01
'''
import scipy.io as scio
import numpy as np


def main():
    data = scio.loadmat('data/outline64x64.mat')
    idx = np.flatnonzero(np.argmax(data['Y'], 1) == 1)
    new_x = np.delete(data['X'], idx, axis=0)
    new_y = np.delete(np.argmax(data['Y'], 1), idx, axis=0)
    new_y = np.array([i > 1 for i in new_y], np.int32)
    new_y = np.eye(2)[new_y]
    print(new_x.shape)
    print(new_y.shape)
    scio.savemat('data/outline64x64-2classes.mat', {"X": new_x, "Y": new_y})


if __name__ == '__main__':
    main()
