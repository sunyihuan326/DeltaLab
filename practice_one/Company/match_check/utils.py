# coding:utf-8
'''
Created on 2017/12/20.

@author: chk01
'''
from aip import AipFace
import numpy as np

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '

client = AipFace(APP_ID, API_KEY, SECRET_KEY)


# 本地图片
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def get_baseInfo(full_path):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark,faceshape,glasses"
    }
    res = client.detect(get_file_content(full_path), options=options)
    result = res['result'][0]
    landmark72 = result['landmark72']

    return landmark72


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]
    return data
