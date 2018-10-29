# coding:utf-8
'''
Created on 2017/11/13.

@author: chk01
'''
import os
import io

import time
import redis
import yaml
from PIL import Image
from pymongo import MongoClient
from urllib.request import urlopen
import numpy as np
import scipy.io as scio

run_mode = os.environ.get('RUN_ENV', 'local')
srv = yaml.load(open('srv.yml', 'r'))[run_mode]
pool = redis.ConnectionPool(**srv['redis'])
rdb = redis.StrictRedis(connection_pool=pool)

mdb = MongoClient(srv['mongo']['host'], srv['mongo']['port'], connect=False, maxPoolSize=50, waitQueueMultiple=10)
mdb.admin.authenticate(srv['mongo']['uname'], str(srv['mongo']['pwd']), mechanism='SCRAM-SHA-1')
mdb = mdb[srv['mongo']['db']]
domain = 'http://dxq.neuling.top'
StyleToNum = {'TMKA': 0, 'MLSS': 1, 'QCJJ': 2,
              'ZRYY': 3, 'GYRM': 4, 'ZXCZ': 5,
              'LMMR': 6, 'HLGY': 7, 'XDMD': 8}
LabelToCode = {
    'TMKA': [1, 0, 0,
             0, 0, 0,
             0, 0, 0],
    'MLSS': [0, 1, 0,
             0, 0, 0,
             0, 0, 0],
    'QCJJ': [0, 0, 1,
             0, 0, 0,
             0, 0, 0],
    'ZRYY': [0, 0, 0,
             1, 0, 0,
             0, 0, 0],
    'GYRM': [0, 0, 0,
             0, 1, 0,
             0, 0, 0],
    'ZXCZ': [0, 0, 0,
             0, 0, 1,
             0, 0, 0],
    'LMMR': [0, 0, 0,
             0, 0, 0,
             1, 0, 0],
    'HLGY': [0, 0, 0,
             0, 0, 0,
             0, 1, 0],
    'XDMD': [0, 0, 0,
             0, 0, 0,
             0, 0, 1],
}
ShapeToCode = {
    'heart': [1, 0, 0, 0, 0],
    'round': [0, 1, 0, 0, 0],
    'oval': [0, 0, 1, 0, 0],
    'square': [0, 0, 0, 1, 0],
    'triangle': [0, 0, 0, 0, 1]
}


def get_rec(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    min_x = min(x)
    min_y = min(y)
    max_x = max(x)
    max_y = max(y)

    wid = max_x - min_x
    hei = max_y - min_y

    max_len = max(wid, hei) + 250
    new_min_x = min_x - (max_len - wid) // 2
    new_max_x = max_x + (max_len - wid) // 2
    new_min_y = min_y - 9 * (max_len - hei) // 10
    new_max_y = max_y + (max_len - hei) // 10

    return (new_min_x, new_min_y, new_max_x, new_max_y)


def load_data():
    tic = time.time()
    sources = mdb.style_source.find({"type": "train"})
    num = sources.count()
    x = np.zeros([num, 64 * 64, 3])
    y = np.zeros([9, 1, num])

    for i, source in enumerate(sources):
        try:
            tic2 = time.time()
            image_bytes = urlopen(domain + source['path']).read()
            data_stream = io.BytesIO(image_bytes)
            pil_image = Image.open(data_stream)
            outline = source['chin']
            eye1 = source['right_eyebrow']
            eye2 = source['left_eyebrow']
            point = []
            point.extend(outline)
            point.extend(eye1)
            point.extend(eye2)
            box = get_rec(point)
            img = pil_image.crop(box)
            img2 = np.array(img.resize([64, 64])).reshape(-1, 3).astype(np.float32)
            image = np.multiply(img2, 1.0 / 255.0)

            x[i, :, :] = image
            y[:, :, i] = np.array(LabelToCode[source['label']]).reshape(9, 1)
            print('图' + str(source['_id']) + '==》cost：', time.time() - tic2)
        except Exception as e:
            print(e)
    scio.savemat('face_data_3_64.mat', {'X': x, 'Y': y})

    print('共耗时：', time.time() - tic)


def load_shape_data():
    tic = time.time()
    sources = mdb.style_source.find({"type": "train"})
    num = sources.count()
    x = np.zeros([num, 64 * 64, 3])
    y = np.zeros([5, 1, num])
    _id = np.zeros([1, 1, num])

    for i, source in enumerate(sources):
        try:
            tic2 = time.time()
            image_bytes = urlopen(domain + source['path']).read()
            data_stream = io.BytesIO(image_bytes)
            pil_image = Image.open(data_stream)
            outline = source['chin']
            eye1 = source['right_eyebrow']
            eye2 = source['left_eyebrow']
            point = []
            point.extend(outline)
            point.extend(eye1)
            point.extend(eye2)
            box = get_rec(point)
            img = pil_image.crop(box)
            img2 = np.array(img.resize([64, 64])).reshape(-1, 3).astype(np.float32)
            image = np.multiply(img2, 1.0 / 255.0)
            _id[:, :, i] = np.array([source['_id']]).reshape(1, 1)
            x[i, :, :] = image
            y[:, :, i] = np.array(ShapeToCode[source['shape']]).reshape(5, 1)
            print('图' + str(source['_id']) + '==》cost：', time.time() - tic2)
        except Exception as e:
            print(e)
    scio.savemat('face_shape_data_3_64.mat', {'X': x, 'Y': y, "ID": _id})

    print('共耗时：', time.time() - tic)


if __name__ == '__main__':
    load_shape_data()
    # data = scio.loadmat('face_data_3_64.mat')
    # print(data['X'].shape)
    # print(data['Y'].shape)
