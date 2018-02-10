# coding:utf-8
from pymongo import MongoClient
import urllib3.request
import urllib
import time
import random

local_mongo = {
    'host': 'dev.yanzijia.cn',
    'port': 3717,
    'uname': 'root',
    'pwd': 'mongo2016sw'
}

mdb_local = MongoClient(local_mongo['host'], local_mongo['port'], connect=False)
mdb_local.admin.authenticate(local_mongo['uname'], local_mongo['pwd'], mechanism='SCRAM-SHA-1')
mdb_local = mdb_local['sheji_dev']


# mdb_release = mdb_local['sheji']


def get_img_up(img_url, fname):
    headers = {'x-gmkerl-rotate': 'auto'}
    if not 'http' in img_url:
        img_url = ('https://xdimg.meiyezhushou.com' + (img_url))
    urllib3.request.urlretriev(img_url, 'D:/1.jpg')
    with open('D:/1.jpg', 'rb') as f:
        upFile.put(fname, f, checksum=True, headers=headers)


def int2hex(num):
    return str(hex(int(num))).replace('0x', '').replace('L', '')


def uniqueName():
    s1 = int2hex(time.time() * 1000)
    s2 = int2hex(random.randint(100000, 999999))
    return s2 + s1


# 五官
def get_xm_sample_face():
    xm_sample_face = mdb_local.xm_sample_face.find({"type": "eye", "status": 100})
    for r in xm_sample_face:
        print(r['_id'])
        del r['_id']
        fname = '/xiaomei/material_library/sample_face/' + uniqueName() + '.jpg'
        get_img_up(r['face_img'], fname)
        r['face_img'] = fname

        if r['type'] in ['face', 'nose', 'eye']:
            for x in range(0, 3):
                for y in range(0, 3):
                    fname = '/xiaomei/material_library/sample_face/' + uniqueName() + '.jpg'
                    get_img_up(r['_' + str(x) + '_' + str(y)], fname)
                    r['_' + str(x) + '_' + str(y)] = fname

        elif r['type'] in ['brow', 'mouth']:
            fname = '/xiaomei/material_library/sample_face/' + uniqueName() + '.jpg'
            get_img_up(r['img'], fname)
            r['img'] = fname

        mdb_release.mdbs.xm_sample_face.insert(r)
        print(r)
        assert 1 == 0


if __name__ == '__main__':
    get_xm_sample_face()
