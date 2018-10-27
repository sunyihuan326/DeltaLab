# coding:utf-8 
'''
created on 2018/10/22

@author:sunyihuan
'''

from PIL import Image
import numpy as np
import os


class ahash(object):

    # 计算hash值
    def getHashCode(self, img, size=(64, 64)):
        pixel = []
        for i in range(size[0]):
            for j in range(size[1]):
                pixel.append(img[i][j])

        mean = sum(pixel) / len(pixel)

        result = []
        for i in pixel:
            if i > mean:
                result.append(1)
            else:
                result.append(0)

        return result
