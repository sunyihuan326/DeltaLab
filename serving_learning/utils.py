# coding:utf-8 
'''
created on 2018/4/2

@author:Dxq
'''
from serving_learning.config import bd_client, tx_client, CIFile
import numpy as np
import urllib.request


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [round(p['x'], 2) for p in points]
    data[:, 1] = [round(p['y'], 2) for p in points]
    return data


def get_baseInfo_bd(full_path):
    '''
    :param full_path: 本地全路径
    :return: dict{"left_eye": array[]*9, "left_eyebrow": array[]*8,
            "face_profile": array[]*13, "nose": array[]*11, "mouth": array[]*14,
            "right_eyebrow": array[]*8, "right_eye": array[]*9}
    '''

    # 本地图片
    def _get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    # 地址图片
    def _get_url_img(filePath):
        image_bytes = urllib.request.urlopen(filePath).read()
        return image_bytes

    options = {
        'max_face_num': 1,
        'face_fields': "landmark"
    }
    res = bd_client.detect(_get_file_content(full_path), options=options)
    result = res['result'][0]
    landmark72 = landmark72_trans(result['landmark72'])

    left_eye = landmark72[13:22]
    left_eyebrow = landmark72[22:30]
    face_profile = landmark72[:13]
    nose = landmark72[47:58]
    mouth = landmark72[58:72]
    right_eyebrow = landmark72[39:47]
    right_eye = landmark72[30:39]

    return {"left_eye": left_eye, "left_eyebrow": left_eyebrow,
            "face_profile": face_profile, "nose": nose, "mouth": mouth,
            "right_eyebrow": right_eyebrow, "right_eye": right_eye}


def get_baseInfo_tx(full_path):
    '''
    :param full_path: 本地全路径
    :return: dict{"left_eye": array[]*8, "left_eyebrow": array[]*8,
            "face_profile": array[]*21, "nose": array[]*13, "mouth": array[]*22,
            "right_eyebrow": array[]*8, "right_eye": array[]*8}
    '''
    data = tx_client.face_detect(CIFile(full_path), mode=1)['data']
    face_data = data['face'][0]
    landmark72 = face_data['face_shape']

    # 五官点
    left_eye = landmark72_trans(landmark72['left_eye'])
    left_eyebrow = landmark72_trans(landmark72['left_eyebrow'])
    face_profile = landmark72_trans(landmark72['face_profile'])
    nose = landmark72_trans(landmark72['nose'])
    mouth = landmark72_trans(landmark72['mouth'])
    right_eyebrow = landmark72_trans(landmark72['right_eyebrow'])
    right_eye = landmark72_trans(landmark72['right_eye'])

    # 脸长宽及左上坐标
    face_width = face_data['width']
    face_height = face_data['height']
    x = face_data['x']
    y = face_data['y']

    gender = face_data['gender']  # 性别
    age = face_data['age']  # 年龄
    hat = face_data['hat']  # 是否带帽子
    glasses = face_data['glasses']  # 是否带眼睛

    roll = face_data['roll']  # 歪头
    pitch = face_data['pitch']  # 低抬头
    yaw = face_data['yaw']  # 侧脸
    mask = face_data['mask']

    expression = face_data['expression']
    beauty = face_data['beauty']
    return {"left_eye": left_eye, "left_eyebrow": left_eyebrow,
            "face_profile": face_profile, "nose": nose, "mouth": mouth,
            "right_eyebrow": right_eyebrow, "right_eye": right_eye,
            "pitch": pitch, "yaw": yaw, "roll": roll, "glasses": glasses, "gender": gender,
            "face_width": face_width, "face_height": face_height, "x": x, "y": y}
