# coding:utf-8
'''
Created on 2017/12/13.

@author: chk01
'''
import numpy as np
import scipy.io as scio
import os


def get_label(file):
    return np.argmax(scio.loadmat('../data/label/' + file)['Label'])


def get_extend_label():
    tt = scio.loadmat('../data/face_1_channel_XY64_expend/label_ids')
    for i in range(9):
        n0 = len(tt['L' + str(i)][0])
        if n0 < 100:
            d0 = np.zeros([2 * n0, 71 * 2])
            dy = np.zeros([2 * n0, 9])
            for k, j in enumerate(tt['L' + str(i)][0]):
                points = scio.loadmat('../data/image3channel/Image-Point-{}.mat'.format(str(j)))['Points']
                X = points
                new_X = X[1:] - X[:-1]
                XX = list(points)
                XX.reverse()
                new_XX = np.array(XX)
                new_XX=new_XX[1:] - new_XX[:-1]
                d0[2 * k, :] = new_X.reshape(1, -1)
                d0[2 * k + 1, :] = new_XX.reshape(1, -1)

                _y = scio.loadmat('../data/label/Image-Label-{}.mat'.format(str(j)))['Label']
                dy[2 * k, :] = _y
                dy[2 * k + 1, :] = _y

        elif n0 > 300:
            d0 = np.zeros([170, 71 * 2])
            dy = np.zeros([170, 9])
            permutation = list(np.random.permutation(170))
            for k, j in enumerate(tt['L' + str(i)][0][permutation]):
                points = scio.loadmat('../data/image3channel/Image-Point-{}.mat'.format(str(j)))['Points']
                X = points
                new_X = X[1:] - X[:-1]
                d0[k, :] = new_X.reshape(1, -1)
                _y = scio.loadmat('../data/label/Image-Label-{}.mat'.format(str(j)))['Label']
                dy[k, :] = _y
        else:
            d0 = np.zeros([n0, 71 * 2])
            dy = np.zeros([n0, 9])
            for k, j in enumerate(tt['L' + str(i)][0]):
                points = scio.loadmat('../data/image3channel/Image-Point-{}.mat'.format(str(j)))['Points']
                X = points
                new_X = X[1:] - X[:-1]
                d0[k, :] = new_X.reshape(1, -1)
                _y = scio.loadmat('../data/label/Image-Label-{}.mat'.format(str(j)))['Label']
                dy[k, :] = _y
        scio.savemat('Label_Point{}'.format(str(i)), {"X": d0, "Y": dy})


def main():
    label_dir = '../data/label'
    files = os.listdir(label_dir)
    data = {
        'L0': [], 'L1': [], 'L2': [], 'L3': [],
        'L4': [], 'L5': [], 'L6': [], 'L7': [],
        'L8': []
    }
    for file in files:
        _id = int(file.replace(".mat", "").split("-")[-1])
        label = get_label(file)
        data['L' + str(label)].append(_id)

    for i in range(9):
        key = 'L' + str(i)
        print(key, '个数：', len(data[key]))
    # L0个数： 36
    # L1个数： 149
    # L2个数： 168
    # L3个数： 81
    # L4个数： 343
    # L5个数： 368
    # L6个数： 34
    # L7个数： 166
    # L8个数： 157
    scio.savemat('../data/face_1_channel_XY64_expend/label_ids', data)


if __name__ == '__main__':
    pass

    # step1
    # main()
    # print(36 + 36 + 149 + 168 + 81 * 2 + 170 + 170 + 34 * 2 + 166 + 157)=1282

    # step2
    # get_extend_label()

    # step3
    data_X = np.zeros((1282, 71 * 2))
    data_Y = np.zeros((1282, 9))
    temp_x = 0
    for i in range(9):
        data = scio.loadmat('Label_Point{}.mat'.format(str(i)))
        n, _ = data['Y'].shape
        data_X[temp_x:temp_x + n, :] = data['X'].reshape(n, -1)
        data_Y[temp_x:temp_x + n, :] = data['Y'].reshape(n, -1)
        temp_x += n
    scio.savemat('face_1_channel_XY_points_expend', {"X": data_X, "Y": data_Y})
