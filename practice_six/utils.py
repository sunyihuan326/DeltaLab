# coding:utf-8
'''
Created on 2017/12/27.

@author: chk01
'''
import os
import numpy as np
import scipy.io as scio

from PIL import Image
from aip import AipFace
import urllib2

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '
client = AipFace(APP_ID, API_KEY, SECRET_KEY)
SHAPE_TRANS = {'oval': "A", 'heart': "B", 'square': "C", 'triangle': "D", 'round': "E"}


class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, Y):
        self.trX = X
        self.trY = Y

    def predict(self, X):
        # distinces = np.linalg.norm(self.trX - X, axis=(1, 2))
        distinces = np.sum(np.linalg.norm(self.trX - X, axis=1), axis=1)
        min_index = np.argmin(np.squeeze(distinces))
        Ypred = np.squeeze(self.trY[min_index])
        return int(Ypred)


# 本地图片
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


# 地址图片
def get_url_img(filePath):
    # image_bytes = urllib.request.urlopen(filePath).read()
    image_bytes = urllib2.urlopen(filePath).read()
    return image_bytes


def load_feature_matrix():
    eyebr_data = scio.loadmat('feature_matrix/eyebr')
    eye_data = scio.loadmat('feature_matrix/eye')
    nose_data = scio.loadmat('feature_matrix/nose')
    lip_data = scio.loadmat('feature_matrix/lip')
    chin_data_A = scio.loadmat('feature_matrix/chin_A')
    chin_data_B = scio.loadmat('feature_matrix/chin_B')
    chin_data_C = scio.loadmat('feature_matrix/chin_C')
    chin_data_D = scio.loadmat('feature_matrix/chin_D')
    chin_data_E = scio.loadmat('feature_matrix/chin_E')

    eyebr = NearestNeighbor()
    eye = NearestNeighbor()
    nose = NearestNeighbor()
    lip = NearestNeighbor()
    chinA = NearestNeighbor()
    chinB = NearestNeighbor()
    chinC = NearestNeighbor()
    chinD = NearestNeighbor()
    chinE = NearestNeighbor()

    eyebr.train(X=eyebr_data['X'], Y=eyebr_data['Y'])
    eye.train(X=eye_data['X'], Y=eye_data['Y'])
    nose.train(X=nose_data['X'], Y=nose_data['Y'])
    lip.train(X=lip_data['X'], Y=lip_data['Y'])

    chinA.train(X=chin_data_A['X'], Y=chin_data_A['Y'])
    chinB.train(X=chin_data_B['X'], Y=chin_data_B['Y'])
    chinC.train(X=chin_data_C['X'], Y=chin_data_C['Y'])
    chinD.train(X=chin_data_D['X'], Y=chin_data_D['Y'])
    chinE.train(X=chin_data_E['X'], Y=chin_data_E['Y'])

    return eyebr, eye, nose, lip, chinA, chinB, chinC, chinD, chinE


def load_cartoon_center():
    return scio.loadmat("feature_matrix/CartoonPoint")


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]
    return data


def get_baseInfo(full_path):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark,faceshape,gender,glasses,qualities"
    }
    res = client.detect(get_file_content(full_path), options=options)
    assert res['result_num'] == 1
    result = res['result'][0]
    assert result['face_probability'] > 0.8

    isPerson = result['qualities']['type']['human']
    isCartoon = result['qualities']['type']['cartoon']
    assert isPerson > isCartoon

    landmark72 = result['landmark72']
    gender = result['gender']
    glasses = result['glasses']
    # faceshape = sorted(result['faceshape'], key=lambda x: -x['probability'])
    # oval,round,square,triangle,heart
    # print(faceshape)
    angle = result['rotation_angle']

    return landmark72, angle, gender, glasses, result['faceshape']


def get_real_faceshape(faceshape):
    _faceshape = sorted(faceshape, key=lambda x: -x['probability'])
    default_shape = _faceshape[0]['type']

    feature_dict = {}
    for i in range(len(faceshape)):
        feature_dict.update({faceshape[i]['type']: faceshape[i]['probability']})

    if feature_dict['triangle'] > 0.45:
        new_shape = 'D'
    elif feature_dict['oval'] > 0.30:
        new_shape = 'A'
    elif feature_dict['heart'] > 0.60:
        new_shape = 'B'
    elif feature_dict['square'] > 0.20:
        new_shape = 'C'
    elif feature_dict['round'] > 0.15:
        new_shape = 'E'
    else:
        new_shape = SHAPE_TRANS[default_shape]

    return new_shape


def point2feature_ebr(landmarks):
    points = landmarks[22:30]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    center = (points[0] + points[4]) / 2
    feature = (points - center) / np.array([wid, hei])
    # feature = (points - center)
    # feature = (points[:-1] - points[1:]) / np.array([wid, hei])
    return feature


def point2feature_eye(landmarks):
    points = landmarks[13:21]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    # center = [(max(x) + min(x)) / 2, (max(y) + min(y)) / 2]
    center = landmarks[17]
    # feature = (points - center) / np.array([wid, hei])
    feature = (points[:-1] - points[1:])
    return feature


def point2feature_nose(landmarks):
    points = landmarks[49:55]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    center = landmarks[57]
    feature = (points - center) / np.array([wid, hei])
    # feature = (points - center)
    return feature


def point2feature_lip(landmarks):
    point1 = np.array([landmarks[58], landmarks[59], landmarks[60], landmarks[61], landmarks[62],
                       landmarks[68], landmarks[67], landmarks[66]])
    x1 = [p[0] for p in point1]
    y1 = [p[1] for p in point1]
    wid1 = max(x1) - min(x1)
    hei1 = max(y1) - min(y1)

    center1 = landmarks[67]
    # feature1 = (point1 - center1) / np.array([wid1, hei1])
    feature1 = (point1[:-1] - point1[1:]) / np.array([wid1, hei1])

    point2 = np.array([landmarks[58], landmarks[65], landmarks[64], landmarks[63], landmarks[62],
                       landmarks[69], landmarks[70], landmarks[71]])
    x2 = [p[0] for p in point2]
    y2 = [p[1] for p in point2]
    wid2 = max(x2) - min(x2)
    hei2 = max(y2) - min(y2)

    center2 = landmarks[70]
    # feature2 = (point2 - center2) / np.array([wid2, hei2])
    feature2 = (point2[:-1] - point2[1:]) / np.array([wid2, hei2])

    feature = np.zeros([14, 2])
    feature[:7, :] = feature1
    feature[7:14, :] = feature2

    # feature[14] = [wid1, hei1]
    # feature[15] = [wid2, hei2]

    return feature


def point2feature_chin(landmarks):
    points = landmarks[:13]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    # landmarks[:13] - landmarks[6]
    return (points - points[6]) / np.array([wid, hei])
