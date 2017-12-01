# coding:utf-8
'''
Created on 2017/11/14.

@author: chk01
'''
import matplotlib.pyplot as plt
import os

import redis
import yaml
from pymongo import MongoClient

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
if __name__ == '__main__':
    sources = mdb.style_source.find({"type": "train"})
    res = {'TMKA': [], 'MLSS': [], 'QCJJ': [],
           'ZRYY': [], 'GYRM': [], 'ZXCZ': [],
           'LMMR': [], 'HLGY': [], 'XDMD': []}
    c_res = {'TMKA': [], 'MLSS': [], 'QCJJ': [],
             'ZRYY': [], 'GYRM': [], 'ZXCZ': [],
             'LMMR': [], 'HLGY': [], 'XDMD': []}
    for source in sources:
        res[source['label']].append(source['area'])
        # y.append(source['eye_dis_ratio'])
        c_res[source['label']].append(StyleToNum[source['label']])
    for i in res.keys():
        plt.scatter(res[i], c_res[i], s=40, c=c_res[i], cmap=plt.cm.Spectral)
    plt.show()
