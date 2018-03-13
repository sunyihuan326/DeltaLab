# coding:utf-8 
'''
created on 2018/3/13

@author:Dxq
'''
import hashlib
import time
import random
import string
from urllib.parse import quote
from PIL import Image
import os
import base64
from urllib.parse import urlencode


def gen_dict_md5(req_dict, app_key):
    # 方法，先对字典排序，排序之后，写app_key，再urlencode
    sort_dict = sorted(req_dict.items(), key=lambda item: item[0], reverse=False)
    # sort_dict=[sort_dict[0],sort_dict[2],sort_dict[3]]
    sort_dict.append(('app_key', app_key))

    sha = hashlib.md5()
    rawtext = urlencode(sort_dict).encode()
    sha.update(rawtext)
    md5text = sha.hexdigest().upper()
    # 字典可以在函数中改写
    if md5text:
        req_dict['sign'] = md5text
    return md5text


def __get_imgfile_base64str__(image):
    if not isinstance(image, str): return None
    if not os.path.isfile(image): return None

    with open(image, 'rb') as fp:
        imgbase64 = base64.b64encode(fp.read())
        return imgbase64


def get_img_base64str(image):
    if isinstance(image, str):
        img_base64str = __get_imgfile_base64str__(image)
    elif isinstance(image, Image):
        img_base64str = __get_imgfile_base64str__(image)
    return img_base64str.decode()


def get_params(plus_item):
    '''''请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效）'''
    t = time.time()
    time_stamp = str(int(t))

    '''''请求随机字符串，用于保证签名不可预测'''
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 10))

    '''''应用标志，这里修改成自己的id和key'''
    app_id = '1106653208'
    app_key = 'qWkUndEe5DTumeQW'

    params = {'app_id': app_id,
              'time_stamp': time_stamp,
              'nonce_str': nonce_str,
              'image': get_img_base64str(plus_item),
              'mode': 0
              }
    sign = gen_dict_md5(params, app_key)
    params['sign'] = sign

    return params
