# coding:utf-8
'''
Created on 2017/11/15.

@author: chk01
'''
import base64
import urllib.request
import io
import time
from aip import AipFace
import scipy.io as scio
import os
import yaml
import redis
from pymongo import MongoClient

run_mode = os.environ.get('RUN_ENV', 'local')
srv = yaml.load(open('srv.yml', 'r'))[run_mode]
pool = redis.ConnectionPool(**srv['redis'])
rdb = redis.StrictRedis(connection_pool=pool)

mdb = MongoClient(srv['mongo']['host'], srv['mongo']['port'], connect=False, maxPoolSize=50, waitQueueMultiple=10)
mdb.admin.authenticate(srv['mongo']['uname'], str(srv['mongo']['pwd']), mechanism='SCRAM-SHA-1')
mdb = mdb[srv['mongo']['db']]
domain = 'http://dxq.neuling.top'
pathDir = 'F:/dataSets/wiki/'

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

data = scio.loadmat('gender.mat')


# 读取图片
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def get_url_img(filePath):
    image_bytes = urllib.request.urlopen(filePath).read()
    return image_bytes


'''
人脸探测
'''


def load_faceshape():
    sources = mdb.style_source.find({"type": "train", "_id": {"$gt": 101359}})
    # 调用人脸属性检测接口
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "faceshape,age"
    }
    for source in sources:
        tic = time.time()
        result = client.detect(get_url_img(domain + source['path']), options=options)
        num = result['result_num']
        if num == 1:
            age = int(result['result'][0]['age'])
            faceshape = result['result'][0]['faceshape']
            faceshape = sorted(faceshape, key=lambda b: -b['probability'])
            mdb.style_source.update_one({"_id": source['_id']}, {"$set": {"age": age, "shape": faceshape[0]['type']}})
            print('更新{}====》OK,cost:{}'.format(source['_id'], time.time() - tic))


if __name__ == '__main__':
    options = {
        'max_face_num': 1,
        'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        # 'face_fields': "faceshape,age"
    }
    result = client.detect(get_file_content('100003.jpg'), options=options)
    print(result['result'][0]['landmark72'])
