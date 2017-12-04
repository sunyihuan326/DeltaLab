# coding:utf-8
'''
Created on 2017/12/2.

@author: chk01
'''
import requests
import pandas as pd
import scipy.io as scio
import numpy as np

dir = 'F:/dataSets/LFPW/testset/'
UseKey = ['_id', 'face_img', 't_outline', 't_sense', 't_style', 't_style_text', 't_face']


# id,图片，轮廓,量感，风格，风格比例，脸型

def fetch_source_data(_id=None):
    if _id:
        _res = requests.post('http://style.yanzijia.cn/ai/user/model', data={'_id': int(_id)}).json()['user']
    else:
        _res = requests.post('http://style.yanzijia.cn/ai/user/model').json()['user']

    res = dict()
    if _res:
        for key in UseKey:
            res.update({key: _res[key]})
    return res


def num_change(idx):
    i = str(idx)
    return '0' * (4 - len(i)) + i


def read_point():
    for i in range(1, 240):
        file_name = 'image_{}.pts'.format(num_change(i))
        print(file_name, 'loding---->------>')
        try:
            openFileHandle = open(dir + file_name, 'r')
        except:
            continue
        j = 0
        tt = []

        while True:
            line = openFileHandle.readline()
            j += 1
            if j > 3:
                if line:
                    point = line.replace('\n', '').split(' ')
                    if len(point) == 2:
                        tt.append(np.array(point).astype('float'))
                else:
                    openFileHandle.close()
                    break
        try:
            assert np.array(tt).shape == (68, 2)
        except:
            print(file_name)
            continue
        scio.savemat(dir + file_name.replace('pts', 'mat'), {'landmarks72': np.array(tt)})
        print(file_name, 'loding---->------>', 'OK')
