# coding:utf-8
'''
Created on 2017/12/13.

@author: chk01
'''
# from aip import AipFace
import urllib.request
import numpy as np
import scipy.io as scio
import os

# """ 你的 APPID AK SK """
# APP_ID = '10365287'
# API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
# SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '
#
# client = AipFace(APP_ID, API_KEY, SECRET_KEY)
# root_dir = 'C:/Users/chk01/Desktop/Delta/image'
# TypOrgans = ['face', 'lip', 'nose', 'left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow']

accept_ans = [
    [0, 1, 3],
    [1, 0, 2, 4],
    [2, 1, 5],
    [3, 0, 4, 6],
    [4, 1, 3, 5, 7],
    [5, 2, 4, 8],
    [6, 3, 7],
    [7, 6, 4, 8],
    [8, 7, 5],
]
low_accept_ans = [
    [0, 1, 3, 4],
    [1, 0, 2, 4, 3, 5],
    [2, 1, 5, 4],
    [3, 0, 4, 6, 1, 7],
    [4, 1, 3, 5, 7, 0, 2, 6, 8],
    [5, 2, 4, 8, 1, 7],
    [6, 3, 7, 4],
    [7, 6, 4, 8, 3, 5],
    [8, 7, 5, 4],
]
absolute_error = [
    [2, 5, 6, 7, 8],
    [6, 7, 8],
    [0, 3, 6, 7, 8],
    [2, 5, 8],
    [0, 2, 6, 8],
    [0, 3, 6],
    [0, 1, 2, 5, 8],
    [0, 1, 2],
    [0, 1, 2, 3, 6]
]
outline_parameters = scio.loadmat('para/outline2.mat')
sense_parameters = scio.loadmat('para/sense5.mat')


def get_area(p1, p2, p3):
    a = np.sqrt(np.sum(np.square(p1 - p2)))
    b = np.sqrt(np.sum(np.square(p1 - p3)))
    c = np.sqrt(np.sum(np.square(p2 - p3)))
    p = (a + b + c) / 2
    area = np.sqrt(p * (p - a) * (p - b) * (p - c))
    return area


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def get_url_img(filePath):
    image_bytes = urllib.request.urlopen(filePath).read()
    return image_bytes


def get_landmark72(full_path):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark"
    }
    result = client.detect(get_file_content(full_path), options=options)
    landmark72 = result['result'][0]['landmark72']
    return landmark72


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]
    return data


def Img2Point(file):
    landmark72 = landmark72_trans(get_landmark72(file))
    points = list(landmark72[:13])
    # points.append(landmark72[21])
    # points.append(landmark72[38])
    return points


def get_outline(points):
    W = outline_parameters['W1']
    b = outline_parameters['b1']
    X = np.array(points).reshape(1, -1)
    Z = np.add(np.matmul(X, W.T), b)
    return np.squeeze(np.argmax(Z))


def get_sense(landmark72):
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

    W = sense_parameters['W1']
    b = sense_parameters['b1']
    Z = np.add(np.matmul(features, W.T), b)
    return np.squeeze(np.argmax(Z))


def sigle_pic(file):
    point = Img2Point(file)
    outline = get_outline(point[:13])
    sense = get_sense(point)
    return '轮廓：=====', outline, '质感：=========', sense


def error2w():
    pass


def main():
    pointdir = '../../practice_two/data/image3channel'
    labeldir = '../../practice_two/data/label'
    files = os.listdir(pointdir)
    error = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
        '4': [],
        '5': [],
        '6': [],
        '7': [],
        '8': []
    }
    cor = 0
    low_cor = 0
    res_list = []
    label_list = []
    for i, file in enumerate(files):
        landmark72 = scio.loadmat(pointdir + '/' + file)['Points']

        outline = get_outline(landmark72[:13])
        sense = get_sense(landmark72)

        res = 3 * int(outline) + int(sense)
        res_list.append(res)

        label = np.squeeze(np.argmax(scio.loadmat(labeldir + '/' + file.replace("Point", "Label"))['Label'], 1))
        label_list.append(label)

        if res in accept_ans[label]:
            cor += 1
        if res in low_accept_ans[label]:
            low_cor += 1
        if res in absolute_error[label]:
            error[str(label)].append(res)

    sample_num = len(files)
    print('可接受:===', round(cor * 100 / sample_num, 2))
    print('低要求可接受:===', round(low_cor * 100 / sample_num, 2))
    print('------------------------------------------------')
    for i in range(9):
        print('{}原则性错误:==='.format(i), round(len(error[str(i)]) * 100 / sample_num, 2))
        print('{}原则性错误占样本:==='.format(i),
              round(len(error[str(i)]) * 100 / (label_list.count(i)), 2))
        print('{}在结果中出现比例:==='.format(i), round(res_list.count(i) * 100 / sample_num, 2))
        print('{}在样本中出现比例:==='.format(i), round(label_list.count(i) * 100 / sample_num, 2))
        print('------------------------------------------------------------------------------')
    print('曲在结果中出现的比例', (res_list.count(0) + res_list.count(1) + res_list.count(2)) / sample_num)
    print('中在结果中出现的比例', (res_list.count(3) + res_list.count(4) + res_list.count(5)) / sample_num)
    print('直在结果中出现的比例', (res_list.count(6) + res_list.count(7) + res_list.count(8)) / sample_num)
    print('--------------------------------------------------------------------------------------------')
    print('曲在样本中出现的比例', (label_list.count(0) + label_list.count(1) + label_list.count(2)) / sample_num)
    print('中在样本中出现的比例', (label_list.count(3) + label_list.count(4) + label_list.count(5)) / sample_num)
    print('直在样本中出现的比例', (label_list.count(6) + label_list.count(7) + label_list.count(8)) / sample_num)
    print('--------------------------------------------------------------------------------------------')
    print('小在结果中出现的比例', (res_list.count(0) + res_list.count(3) + res_list.count(6)) / sample_num)
    print('中在结果中出现的比例', (res_list.count(1) + res_list.count(4) + res_list.count(7)) / sample_num)
    print('大在结果中出现的比例', (res_list.count(2) + res_list.count(5) + res_list.count(8)) / sample_num)
    print('--------------------------------------------------------------------------------------------')
    print('小在样本中出现的比例', (label_list.count(0) + label_list.count(3) + label_list.count(6)) / sample_num)
    print('中在样本中出现的比例', (label_list.count(1) + label_list.count(4) + label_list.count(7)) / sample_num)
    print('大在样本中出现的比例', (label_list.count(2) + label_list.count(5) + label_list.count(8)) / sample_num)
    # print('总的原则性错误===', round(sum(error.values()) * 100 / len(files), 2))
    return True


if __name__ == '__main__':
    main()
    # 0的比例 2.41 %
    # 1的比例 9.49 %
    # 2的比例 11.24 %
    # 3的比例 5.75 %
    # 4的比例 23.48 %
    # 5的比例 24.65 %
    # 6的比例 2.25 %
    # 7的比例 10.07 %
    # 8的比例 10.66 %
