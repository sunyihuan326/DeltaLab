# coding:utf-8 
'''
created on 2018/3/13

@author:Dxq
'''
import requests
from Tencent import utils


def get_content(plus_item):
    url = "https://api.ai.qq.com/fcgi-bin/face/face_detectface"  # 聊天的API地址
    payload = utils.get_params(plus_item)  # 获取请求参数
    r = requests.post(url, data=payload)
    return r.json()["data"]


if __name__ == '__main__':
    file = '3.jpg'

    answer = get_content(file)
    print(answer)
