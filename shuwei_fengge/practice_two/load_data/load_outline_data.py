# coding:utf-8
'''
Created on 2017/12/7.

@author: chk01
'''
import os
from PIL import Image, ImageDraw, ImageEnhance
from shuwei_fengge.practice_two.load_data.utils import *

LabelToSense = [0, 1, 2, 0, 1, 2, 0, 1, 2]
LabelToOutline = [0, 0, 0, 1, 1, 1, 2, 2, 2]


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
    # region.show()
    return region


def main(typ):
    logdir = '../data/image3channel'
    label_dir = '../data/label'
    files = os.listdir(logdir)
    num = len(files)
    print('total num ------>', num)
    data_X = np.zeros((num, 13 * 2))
    data_Y = np.zeros((num, 3))
    for i, file in enumerate(files):
        print('read_{}_data------->loading----->start'.format(file))
        points = scio.loadmat(logdir + '/' + file)['Points']
        tt = points[:13] - points[6]

        data_X[i, :] = np.array(tt).reshape(1, -1)
        label = np.argmax(scio.loadmat(label_dir + '/' + file.replace('Point', 'Label'))['Label'])
        if typ == 'outline':
            data_Y[i, :] = convert_to_one_hot(LabelToOutline[label], 3)
        else:
            data_Y[i, :] = convert_to_one_hot(LabelToSense[label], 3)
        print('read_{}_data------->loading----->end'.format(file))

    scio.savemat('../data/outline/face_1_channel_vec{}'.format(typ), {"X": data_X, "Y": data_Y})


if __name__ == '__main__':
    main('outline')
    # pass
    # tt = scio.loadmat('../data/outline/face_1_channel_sense.mat')
    # print(tt['X'].shape)
    # print(tt['Y'].shape)
