# coding:utf-8
'''
Created on 2018/1/2.

@author: chk01
'''

import requests
import base64
import hashlib
import time
import random
import os, string
from PIL import Image
from urllib.parse import urlencode


class MsgTencent(object):
    def __init__(self, AppID=None, AppKey=None):
        '''
        改成你自己的API账号、密码
        '''
        if not AppID: AppID = '1106653208'
        if not AppKey: AppKey = 'qWkUndEe5DTumeQW'
        self.app_id = AppID
        self.app_key = AppKey
        self.img_base64str = None

    def get_random_str(self):
        # 随机生成16位字符串
        rule = string.ascii_lowercase + string.digits
        str = random.sample(rule, 16)
        return "".join(str)

    def get_time_stamp(self):
        return str(int(time.time()))

    # def __get_image_base64str__(self, image):
    #     if not isinstance(image, Image):
    #         return None
    #     outputBuffer = BytesIO()
    #     bg.save(outputBuffer, format='JPEG')
    #     imgbase64 = base64.b64encode(outputBuffer.getvalue())
    #     return imgbase64

    def __get_imgfile_base64str__(self, image):
        if not isinstance(image, str): return None
        if not os.path.isfile(image): return None

        with open(image, 'rb') as fp:
            imgbase64 = base64.b64encode(fp.read())
            return imgbase64

    def get_img_base64str(self, image):
        if isinstance(image, str):
            self.img_base64str = self.__get_imgfile_base64str__(image)
        elif isinstance(image, Image):
            self.img_base64str = self.__get_imgfile_base64str__(image)
        return self.img_base64str.decode()

        # 组装字典，MD5加密方法

    '''
    ======================================
    tencent获得参数对列表N（字典升级排序）
    ======================================
    1\依照算法第一步要求，对参数对进行排序，得到参数对列表N如下。
    参数名     参数值
    app_id     10000
    nonce_str     20e3408a79
    text     腾讯开放平台
    time_stamp     1493449657

    2\按URL键值拼接字符串T
    依照算法第二步要求，将参数对列表N的参数对进行URL键值拼接，值使用URL编码，URL编码算法用大写字母，例如%E8，而不是小写%e8，得到字符串T如下：
    app_id=10000&nonce_str=20e3408a79&text=%E8%85%BE%E8%AE%AF%E5%BC%80%E6%94%BE%E5%B9%B3%E5%8F%B0&time_stamp=1493449657

    3\拼接应用密钥，得到字符串S
    依照算法第三步要求，将应用密钥拼接到字符串T的尾末，得到字符串S如下。
    app_id=10000&nonce_str=20e3408a79&text=%E8%85%BE%E8%AE%AF%E5%BC%80%E6%94%BE%E5%B9%B3%E5%8F%B0&time_stamp=1493449657&app_key=a95eceb1ac8c24ee28b70f7dbba912bf

    4\计算MD5摘要，得到签名字符串
    依照算法第四步要求，对字符串S进行MD5摘要计算得到签名字符串如。
    e8f6f347d549fe514f0c9c452c95da9d

    5\转化md5签名值大写
    对签名字符串所有字母进行大写转换，得到接口请求签名，结束算法。
    E8F6F347D549FE514F0C9C452C95DA9D

    6\最终请求数据
    在完成签名计算后，即可得到所有接口请求数据，进一步完成API的调用。
    text     腾讯开放平台     接口请求数据，UTF-8编码
    app_id     10000     应用标识
    time_stamp     1493449657     请求时间戳（秒级），用于防止请求重放
    nonce_str     20e3408a79     请求随机字符串，用于保证签名不可预测
    sign     E8F6F347D549FE514F0C9C452C95DA9D     请求签名    
    '''

    def gen_dict_md5(self, req_dict, app_key):
        if not isinstance(req_dict, dict): return None
        if not isinstance(app_key, str) or not app_key: return None
        try:
            # 方法，先对字典排序，排序之后，写app_key，再urlencode
            sort_dict = sorted(req_dict.items(), key=lambda item: item[0], reverse=False)
            # sort_dict=[sort_dict[0],sort_dict[2],sort_dict[3]]
            sort_dict.append(('app_key', app_key))

            sha = hashlib.md5()
            rawtext = urlencode(sort_dict).encode()
            sha.update(rawtext)
            md5text = sha.hexdigest().upper()
            # 字典可以在函数中改写
            if md5text: req_dict['sign'] = md5text
            return md5text
        except Exception as e:
            print('error')
            return None

    # 生成字典
    def gen_req_dict(self, req_dict, app_id=None, app_key=None, time_stamp=None, nonce_str=None):
        """用MD5算法生成安全签名"""
        if not req_dict.get('app_id'):
            if not app_id: app_id = self.app_id
            req_dict['app_id'] = app_id

        # nonce_str 字典无值
        if not req_dict.get('time_stamp'):
            if not time_stamp: time_stamp = self.get_time_stamp()
            req_dict['time_stamp'] = time_stamp

        if not req_dict.get('nonce_str'):
            if not nonce_str: nonce_str = self.get_random_str()
            req_dict['nonce_str'] = nonce_str
        # app_key 取系统参数。
        if not app_key: app_key = self.app_key

        md5key = self.gen_dict_md5(req_dict, app_key)
        return md5key


TencentAPI = {
    "face_dect": {
        'APINAME': '人脸识别',
        'APIDESC': '识别上传图像上面的人脸信息',
        'APIURL': 'https://api.ai.qq.com/fcgi-bin/face/face_detectface',
        'APIPARA': 'image'
    },
}


def ExecTecentAPI(*arg, **kwds):
    if kwds.get('Apiname'):
        apiname = kwds.pop('Apiname')

    url = TencentAPI[apiname]['APIURL']
    # name = TencentAPI[apiname]['APINAME']
    # desc = TencentAPI[apiname]['APIDESC']
    para = TencentAPI[apiname]['APIPARA']

    tx = MsgTencent()

    Req_Dict = {}
    for key in para.split(','):
        value = None
        if kwds.get(key):  value = kwds.pop(key)
        if key == 'image':
            # 图像获取base64
            value = tx.get_img_base64str(value)
        if key == 'text':
            # 文本进行GBK编码
            value = value.encode('gbk')

        Req_Dict[key] = value
        # print(key, value, Req_Dict[key])
    # 生成请求包
    Req_Dict['mode'] = 1
    sign = tx.gen_req_dict(req_dict=Req_Dict)
    resp = requests.post(url, data=Req_Dict)
    return resp.text


if __name__ == "__main__":
    file = '1002.jpg'
    rest = ExecTecentAPI(Apiname='face_dect', image=file)
    landmark72 = rest['data']['face_list'][0]['face_shape']
    for key in ['face_profile', '', '', '', '', '', '', '', '', '', '']:
        pass
    print(rest)
    # 文本分析
    # rest = ExecTecentAPI(Apiname='nlp_wordseg', text='上帝保佑你')
