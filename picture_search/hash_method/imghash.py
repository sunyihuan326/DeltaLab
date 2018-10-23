# coding:utf-8 
'''
created on 2018/10/16

@author:sunyihuan
'''
from PIL import Image
import numpy as np
import os
import hashlib
import struct


class imghash(object):
    def getHashCode(self, img):
        data = img.tobytes()
        h = hashlib.sha512()
        h.update(data)
        return h.hexdigest()
