# coding:utf-8
'''
Created on 2018/1/11.

@author: chk01
'''
import requests
from PIL import Image
from io import BytesIO


def get_eval_res(status, reason):
    # 好405
    # 共1020
    url = 'http://xiaomei.meiyezhushou.com/api/m/sample/user/syn/get/list?status={}&&reason={}'.format(status, reason)
    results = requests.get(url).json()
    # 0脸型不像 344
    # 1眉毛不像 85
    # 2眼睛不像 105
    # 3鼻子不像 60
    # 4嘴唇不像 66
    # 5中庭过短 245
    # 6下庭过长 65
    # 7人中过长 133
    # 8人中过短 8
    for i, res in enumerate(results['user_list']):
        print(i)
        response = requests.get(res['img'])
        image = Image.open(BytesIO(response.content))
        image.save('img/{}/{}.png'.format(reason, i))
        # print(res['img'])
        # print(res['img_syn'])


# for ii in range(5):
get_eval_res(0, 5)
