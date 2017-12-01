# coding:utf-8
'''
Created on 2017/6/25.

@author: Dxq
'''
import os
import time

import yaml
import redis
from pymongo import MongoClient

ospath = os.path.split(__file__)[0]

run_mode = os.environ.get('RUN_ENV', 'local')

gconf = yaml.load(open(ospath + '/gconf.yml', 'r'))[run_mode]

gconf['uptime'] = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

srv = yaml.load(open(ospath + '/srv.yml', 'r'))[run_mode]

errorDesc = yaml.load(open(ospath + '/errorDesc.yml', 'r'))

msgDict = {}

pool = redis.ConnectionPool(**srv['redis'])
rdb = redis.StrictRedis(connection_pool=pool)

mdb = MongoClient(srv['mongo']['host'], srv['mongo']['port'], connect=False, maxPoolSize=50, waitQueueMultiple=10)
mdb.admin.authenticate(srv['mongo']['uname'], str(srv['mongo']['pwd']), mechanism='SCRAM-SHA-1')
mdb = mdb[srv['mongo']['db']]
