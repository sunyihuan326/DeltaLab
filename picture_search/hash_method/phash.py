# coding:utf-8 
'''
created on 2018/10/22

@author:sunyihuan
'''
import math
from PIL import Image
import numpy as np
import os


class phash(object):

    # 计算系数矩阵
    def getCoefficient(self, length):
        C_temp = np.zeros((length, length))
        C_temp[0, :] = 1 * np.sqrt(1 / length)

        for i in range(1, length):
            for j in range(length):
                C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * length)
                                      ) * np.sqrt(2 / length)
        return C_temp

    # 计算DCT
    def DCT(self, matrix):
        length, length1 = matrix.shape
        C_temp = self.getCoefficient(length)
        dst = np.dot(C_temp, matrix)
        DCT_matrix = np.dot(dst, np.transpose(C_temp))
        return DCT_matrix

    # 计算左上角8*8并转化为list
    def submatrix_list(self, matrix, size=(8, 8)):
        matrix = self.DCT(matrix)
        value = []
        for i in range(size[0]):
            for j in range(size[1]):
                value.append(matrix[i][j])
        return value

    # 计算hash值
    def getHashCode(self, img):
        sub_list = self.submatrix_list(img)
        length = len(sub_list)
        mean = sum(sub_list) / length

        result = []
        for i in sub_list:
            if i > mean:
                result.append(1)
            else:
                result.append(0)

        return result
