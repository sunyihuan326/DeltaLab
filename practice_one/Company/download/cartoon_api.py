# coding:utf-8
'''
Created on 2018/1/24.

@author: chk01
'''
import requests
from practice_one.Company.load_material.utils import *
import base64
from io import BytesIO

domain = 'https://devai.meiyezhushou.com'


def main(file):
    landmark72, angle, gender, glasses, faceshape = get_baseInfo(file)
    # # # 图片矫正待优化
    if -10 < angle < 10:
        pass
    else:
        # import math
        print(angle)
        # angle = -angle / 180 * math.pi
        Image.open(file).rotate(angle, expand=1).save(file)
        landmark72, angle, gender, glasses, faceshape = get_baseInfo(file)
    res = requests.get(
        domain + '/cartoon/merge?landmark72={}&&glasses={}&&faceshape={}&&skin_color={}'.format(landmark72, glasses,
                                                                                                faceshape, 1),
        cert=False).json()

    imgdata = base64.b64decode(str(res['cartoon_face'])[2:-1])
    tt = Image.open(BytesIO(imgdata))
    tt.save('res.png')
    tt.show()
    # local_url = 'res.png'
    # file = open(local_url, 'wb+')
    # file.write(imgdata)
    # file.close()


if __name__ == "__main__":
    i = 2
    file = 'check/{}.jpg'.format(i)
    main(file)
