# coding:utf-8
'''
Created on 2018/1/4.

@author: chk01
'''
import os
from PIL import Image, ImageDraw, ImageEnhance
from practice_two.load_data.utils import *

LabelToSense = [0, 1, 2, 0, 1, 2, 0, 1, 2]


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


def get_area(p1, p2, p3):
    a = np.sqrt(np.sum(np.square(p1 - p2)))
    b = np.sqrt(np.sum(np.square(p1 - p3)))
    c = np.sqrt(np.sum(np.square(p2 - p3)))
    p = (a + b + c) / 2
    area = np.sqrt(p * (p - a) * (p - b) * (p - c))
    return area


def get_senseNorm28(landmark72):
    points = landmark72[:13]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)

    p = points / np.array([wid, hei])
    p = p.reshape(1, -1)

    p0 = landmark72[0]
    p3 = landmark72[3]
    p6 = landmark72[6]
    p9 = landmark72[9]
    p12 = landmark72[12]
    p21 = landmark72[21]
    p38 = landmark72[38]
    p57 = landmark72[57]

    area1 = get_area(p0, p3, p6)
    area2 = get_area(p3, p6, p9)
    area3 = get_area(p6, p9, p12)
    area4 = get_area(p21, p38, p57)
    area5 = get_area(p0, p6, p12)

    features = np.zeros([1, 28])
    features[:, :26] = p
    features[:, 26] = area4 / (area1 + area3 + area5)
    features[:, 27] = area2 / (area1 + area3 + area5)
    return features


def get_sense28(landmark72):
    p = landmark72[:13].reshape(1, -1)
    p0 = landmark72[0]
    p3 = landmark72[3]
    p6 = landmark72[6]
    p9 = landmark72[9]
    p12 = landmark72[12]
    p21 = landmark72[21]
    p38 = landmark72[38]
    p57 = landmark72[57]

    area1 = get_area(p0, p3, p6)
    area2 = get_area(p3, p6, p9)
    area3 = get_area(p6, p9, p12)
    area4 = get_area(p21, p38, p57)
    area5 = get_area(p0, p6, p12)

    features = np.zeros([1, 28])
    features[:, :26] = p
    features[:, 26] = area4 / (area1 + area3 + area5)
    features[:, 27] = area2 / (area1 + area3 + area5)
    return features


def get_sense26(landmark72):
    points = landmark72[:13]
    feature = (points - points[6])
    p = np.concatenate([feature[:6], feature[7:]], axis=0).reshape(1, -1)

    p0 = landmark72[0]
    p3 = landmark72[3]
    p6 = landmark72[6]
    p9 = landmark72[9]
    p12 = landmark72[12]
    p21 = landmark72[21]
    p38 = landmark72[38]
    p57 = landmark72[57]

    area1 = get_area(p0, p3, p6)
    area2 = get_area(p3, p6, p9)
    area3 = get_area(p6, p9, p12)
    area4 = get_area(p21, p38, p57)
    area5 = get_area(p0, p6, p12)

    features = np.zeros([1, 26])
    features[:, :24] = p
    features[:, 24] = area4 / (area1 + area3 + area5)
    features[:, 25] = area2 / (area1 + area3 + area5)
    return features


def get_senseNorm26(landmark72):
    points = landmark72[:13]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)

    feature = (points - points[6]) / np.array([wid, hei])
    p = np.concatenate([feature[:6], feature[7:]], axis=0).reshape(1, -1)

    p0 = landmark72[0]
    p3 = landmark72[3]
    p6 = landmark72[6]
    p9 = landmark72[9]
    p12 = landmark72[12]
    p21 = landmark72[21]
    p38 = landmark72[38]
    p57 = landmark72[57]

    area1 = get_area(p0, p3, p6)
    area2 = get_area(p3, p6, p9)
    area3 = get_area(p6, p9, p12)
    area4 = get_area(p21, p38, p57)
    area5 = get_area(p0, p6, p12)

    features = np.zeros([1, 26])
    features[:, :24] = p
    features[:, 24] = area4 / (area1 + area3 + area5)
    features[:, 25] = area2 / (area1 + area3 + area5)
    return features


def main():
    logdir = '../../practice_two/data/image3channel'
    label_dir = '../../practice_two/data/label'
    files = os.listdir(logdir)
    num = len(files)
    print('total num ------>', num)
    data_X = np.zeros((num, 26))
    data_Y = np.zeros((num, 3))
    for i, file in enumerate(files):
        print('read_{}_data------->loading----->start'.format(file))
        points = scio.loadmat(logdir + '/' + file)['Points']
        data_X[i, :] = get_senseNorm26(points)
        label = np.argmax(scio.loadmat(label_dir + '/' + file.replace('Point', 'Label'))['Label'])
        data_Y[i, :] = convert_to_one_hot(LabelToSense[label], 3)

        print('read_{}_data------->loading----->end'.format(file))

    scio.savemat('data/senseNorm26', {"X": data_X, "Y": data_Y})


if __name__ == '__main__':
    main()
