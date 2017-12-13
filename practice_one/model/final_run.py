# coding:utf-8
'''
Created on 2017/12/13.

@author: chk01
'''
from aip import AipFace
import urllib.request
import numpy as np
import scipy.io as scio
import os

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '

client = AipFace(APP_ID, API_KEY, SECRET_KEY)
root_dir = 'C:/Users/chk01/Desktop/Delta/image'
TypOrgans = ['face', 'lip', 'nose', 'left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow']
PointNum = {
    'face': 13,
    'lip': 14,
    'nose': 11,
    'left_eye': 9,
    'right_eye': 9,
    'left_eyebrow': 8,
    'right_eyebrow': 8
}
outline_parameters = scio.loadmat('para/outline.mat')
sense_parameters = scio.loadmat('para/sense.mat')


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
    points.append(landmark72[21])
    points.append(landmark72[38])
    return points


def get_outline(points):
    W = outline_parameters['W1']
    b = outline_parameters['b1']
    X = np.array(points).reshape(1, -1)
    Z = np.add(np.matmul(X, W.T), b)
    return np.squeeze(np.argmax(Z))


def get_sense(points):
    W = sense_parameters['W1']
    b = sense_parameters['b1']
    X = np.array(points).reshape(1, -1)
    Z = np.add(np.matmul(X, W.T), b)
    return np.squeeze(np.argmax(Z))


if __name__ == '__main__':
    pointdir = '../../practice_two/data/image3channel'
    files = os.listdir(pointdir)
    data = []
    for i, file in enumerate(files):
        landmark72 = scio.loadmat(pointdir + '/' + file)['Points']
        points = list(landmark72[:13])
        points.append(landmark72[21])
        points.append(landmark72[38])

        outline = get_outline(points[:13])
        sense = get_sense(points)
        res = 3 * int(sense) + int(outline)
        data.append(res)

    for i in range(9):
        print(i, round(data.count(i) * 100 / len(data), 2))
