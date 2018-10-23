# coding:utf-8 
'''
created on 2018/10/22

@author:sunyihuan
'''
from PIL import Image
import numpy as np
import os


class dhash(object):
    # 计算hash值
    def getHashCode(self, img, size=(8, 8)):
        result = []
        for i in range(size[0] - 1):
            for j in range(size[1]):
                current_val = img[i][j]
                next_val = img[i + 1][j]
                if current_val > next_val:
                    result.append(1)
                else:
                    result.append(0)

        return result
