# coding:utf-8
'''
Created on 2017/12/13.

@author: chk01
'''
import os
import scipy.io as scio
import numpy as np


def main():
    logdir = '../data/image3channel'
    label_dir = '../data/label'
    files = os.listdir(logdir)
    num = len(files)
    print('total num ------>', num)
    data_X = np.zeros((num, 71 * 2))
    data_Y = np.zeros((num, 9))
    for i, file in enumerate(files):
        print('read_{}_data------->loading----->start'.format(file))
        points = scio.loadmat(logdir + '/' + file)['Points']
        p2v = points[1:] - points[:-1]

        data_X[i, :] = np.array(p2v).reshape(1, -1)
        data_Y[i, :] = scio.loadmat(label_dir + '/' + file.replace('Point', 'Label'))['Label']
        print('read_{}_data------->loading----->end'.format(file))

    scio.savemat('face_1_channel_XY_Points', {"X": data_X, "Y": data_Y})


if __name__ == '__main__':
    tt = scio.loadmat('face_1_channel_XY_Points.mat')
    print(tt['X'].shape)
    # main()
