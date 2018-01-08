# coding:utf-8
'''
Created on 2018/1/6.

@author: chk01
'''
from practice_five.utils import *
from PIL import Image
from io import BytesIO
import requests
import scipy.io as scio
import matplotlib.pyplot as plt
import os

_resList = requests.get('http://xiaomei.meiyezhushou.com/api/m/sample/brow/list/all').json()['brow_list']
m = len(_resList)

trX = np.zeros([m, 64 * 64 * 3])
trY = np.zeros([m, 18])
for i, res in enumerate(_resList):
    # if not os.path.exists('data_crash/{}.jpg'.format(res['_id'])):
    # plt.figure(i + 1)
    # print('----load----start---', res['_id'])
    img = Image.open(BytesIO(requests.get(res['img']).content))
    X = np.array(img).reshape(1, -1)
    p1 = [res['ll_x'], res['ll_y']]
    p2 = [res['llu_x'], res['llu_y']]
    p3 = [res['lmu_x'], res['lmu_y']]
    p4 = [res['lru_x'], res['lru_y']]
    p5 = [res['mv_x'], res['mv_y']]
    p6 = [res['rr_x'], res['rr_y']]
    p7 = [res['rru_x'], res['rru_y']]
    p8 = [res['rmu_x'], res['rmu_y']]
    p9 = [res['rlu_x'], res['rlu_y']]
    pp = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9])
    plt.scatter(pp[:, 0], -pp[:, 1])
    if i % 20 == 0:
        plt.show()
    # plt.savefig('data_crash/{}.jpg'.format(res['_id']))
# Y = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9]).reshape(1, -1)
#     trX[i, :] = X
#     trY[i, :] = Y
#     print('----load----end---', i)
#
# scio.savemat('data/face_top_9', {"X": trX, "Y": trY})
