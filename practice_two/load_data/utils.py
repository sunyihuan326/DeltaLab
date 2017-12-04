# coding:utf-8
'''
Created on 2017/12/2.

@author: chk01
'''
import requests
import pandas as pd
import scipy.io as scio
import numpy as np
from aip import AipFace
import urllib.request
import matplotlib.pyplot as plt

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '

client = AipFace(APP_ID, API_KEY, SECRET_KEY)
UseKey = ['_id', 'face_img', 't_outline', 't_sense', 't_style', 't_style_text', 't_face']


# id,图片，轮廓,量感，风格，风格比例，脸型

def fetch_source_data(_id=None):
    if _id:
        _resList = [requests.post('http://style.yanzijia.cn/ai/user/model', data={'_id': int(_id)}).json()['user']]
    else:
        _resList = requests.post('http://style.yanzijia.cn/ai/user/model').json()['user']

    resList = list()
    if _resList:
        for _res in _resList:
            res = dict()
            for key in UseKey:
                res.update({key: _res[key]})
            resList.append(res)

    return resList


def num_change(idx):
    i = str(idx)
    return '0' * (4 - len(i)) + i


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def get_url_img(filePath):
    image_bytes = urllib.request.urlopen(filePath).read()
    return image_bytes


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]
    return data


def get_landmark72(full_path, typ='local'):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark"
    }
    if typ == 'local':
        result = client.detect(get_file_content(full_path), options=options)
    else:
        result = client.detect(get_url_img(full_path), options=options)

    landmark72 = landmark72_trans(result['result'][0]['landmark72'])

    return landmark72


def draw_pic(data, index=1):
    # plt.subplot(3, 2, index)
    # x = [d['x'] for d in data]
    # y = [-d['y'] for d in data]
    plt.scatter(data[:, 0], -data[:, 1])
    plt.show()
    # ax = plt.gca()
    # ax.set_aspect(1)

# dir = 'F:/dataSets/LFPW/trainset/'
# ids = 34
# draw_pic(get_landmark72(dir + 'image_00{}.png'.format(ids)))
# dd = scio.loadmat(dir + 'image_00{}.mat'.format(ids))
# draw_pic(dd['landmarks72'])
