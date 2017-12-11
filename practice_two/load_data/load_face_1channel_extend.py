# coding:utf-8
'''
Created on 2017/12/4.

@author: chk01
'''
import scipy.io as scio
import os
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np


def get_label(file):
    return np.argmax(scio.loadmat('../data/label/' + file)['Label'])


def get_face_box(points):
    X = points[:, 0]
    Y = points[:, 1]
    min_x = min(X) - 10
    max_x = max(X) + 10
    min_y = min(Y) - 10
    max_y = max(Y) + 10

    wid = max(max_y - min_y, max_x - min_x)

    new_x = min_x - (wid - (max_x - min_x)) // 2
    new_y = min_y - (wid - (max_y - min_y)) // 2

    pil_image = Image.new("RGB", (2000, 2000), color=255)
    d = ImageDraw.Draw(pil_image)

    d.line([tuple(p) for p in points[:13]], width=10, fill=0)
    d.line([tuple(p) for p in points[13:21]], width=10, fill=0)
    d.line([tuple(p) for p in [points[13], points[20]]], width=10, fill=0)
    d.line([tuple(p) for p in points[30:38]], width=10, fill=0)
    d.line([tuple(p) for p in [points[30], points[37]]], width=10, fill=0)
    d.line([tuple(p) for p in points[22:30]], width=10, fill=0)
    d.line([tuple(p) for p in [points[22], points[29]]], width=10, fill=0)
    d.line([tuple(p) for p in points[39:47]], width=10, fill=0)
    d.line([tuple(p) for p in [points[39], points[46]]], width=10, fill=0)
    d.line([tuple(p) for p in points[47:57]], width=10, fill=0)
    d.line([tuple(p) for p in points[58:66]], width=10, fill=0)
    d.line([tuple(p) for p in [points[58], points[65]]], width=10, fill=0)

    region = pil_image.crop([new_x, new_y, new_x + wid, new_y + wid])
    region = region.resize((64, 64), Image.ANTIALIAS).convert("L")
    region = ImageEnhance.Contrast(region).enhance(999)
    return region


def get_extend_label():
    tt = scio.loadmat('label_ids.mat')
    for i in range(9):
        n0 = len(tt['L' + str(i)][0])
        if n0 < 100:
            d0 = np.zeros([2 * n0, 64, 64])
            dy = np.zeros([2 * n0, 9])
            for k, j in enumerate(tt['L' + str(i)][0]):
                points = scio.loadmat('../data/image3channel/Image-Point-{}.mat'.format(str(j)))['Points']
                X = get_face_box(points)
                XX = X.transpose(Image.FLIP_LEFT_RIGHT)
                d0[2 * k, :, :] = X
                d0[2 * k + 1, :, :] = XX

                _y = scio.loadmat('../data/label/Image-Label-{}.mat'.format(str(j)))['Label']
                dy[2 * k, :] = _y
                dy[2 * k + 1, :] = _y

        elif n0 > 300:
            d0 = np.zeros([200, 64, 64])
            dy = np.zeros([200, 9])
            permutation = list(np.random.permutation(200))
            for k, j in enumerate(tt['L' + str(i)][0][permutation]):
                points = scio.loadmat('../data/image3channel/Image-Point-{}.mat'.format(str(j)))['Points']
                X = get_face_box(points)
                d0[k, :, :] = X
                _y = scio.loadmat('../data/label/Image-Label-{}.mat'.format(str(j)))['Label']
                dy[k, :] = _y
        else:
            d0 = np.zeros([n0, 64, 64])
            dy = np.zeros([n0, 9])
            for k, j in enumerate(tt['L' + str(i)][0]):
                points = scio.loadmat('../data/image3channel/Image-Point-{}.mat'.format(str(j)))['Points']
                X = get_face_box(points)
                d0[k, :, :] = X
                _y = scio.loadmat('../data/label/Image-Label-{}.mat'.format(str(j)))['Label']
                dy[k, :] = _y
        scio.savemat('Label{}'.format(str(i)), {"X": d0, "Y": dy})


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
    scio.savemat('label_ids', data)


# 1341
if __name__ == '__main__':
    # get_extend_label()
    # main()
    data_X = np.zeros((1342, 64 * 64))
    data_Y = np.zeros((1342, 9))
    temp_x = 0

    for i in range(9):
        data = scio.loadmat('Label{}.mat'.format(str(i)))
        n, _ = data['Y'].shape
        data_X[temp_x:temp_x + n, :] = data['X'].reshape(n, -1)
        data_Y[temp_x:temp_x + n, :] = data['Y'].reshape(n, -1)
        temp_x += n
    scio.savemat('res', {"X": data_X, "Y": data_Y})
