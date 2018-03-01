# coding:utf-8 
'''
created on 2018/2/28

@author:Dxq
'''
import numpy as np
from aip import AipFace
import urllib.request

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '
client = AipFace(APP_ID, API_KEY, SECRET_KEY)


# 本地图片
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


# 地址图片
def get_url_img(filePath):
    image_bytes = urllib.request.urlopen(filePath).read()
    return image_bytes


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]

    res = dict()
    res['chin'] = data[:13]
    res['left_eye'] = data[13:22]
    res['right_eye'] = data[30:39]
    res['left_brow'] = data[22:30]
    res['right_brow'] = data[39:47]
    res['nose'] = data[47:58]
    res['lip'] = data[58:72]

    return data, res


def get_baseInfo(full_path):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark,faceshape,gender,glasses,qualities"
    }
    res = client.detect(get_file_content(full_path), options=options)
    assert res['result_num'] == 1
    result = res['result'][0]
    # assert result['face_probability'] > 0.8

    isPerson = result['qualities']['type']['human']
    isCartoon = result['qualities']['type']['cartoon']
    # assert isPerson > isCartoon

    landmark72_list, landmark72_dict = landmark72_trans(result['landmark72'])
    # gender = result['gender']
    # glasses = result['glasses']

    angle = result['rotation_angle']

    return landmark72_list, landmark72_dict, angle
