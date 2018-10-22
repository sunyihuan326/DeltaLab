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
    def __init__(self, img1_path, img2_path):
        self.img1 = Image.open(img1_path)
        self.img2 = Image.open(img2_path)

    # 正则化图像
    def regularizeImage(self, img, size=(256, 256)):
        return np.array(img.resize(size).convert('L'))

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
        value = []
        for i in range(size[0]):
            for j in range(size[1]):
                value.append(matrix[i][j])
        return value

    # 计算hash值
    def getHashCode(self, sub_list):
        length = len(sub_list)
        mean = sum(sub_list) / length

        result = []
        for i in sub_list:
            if i > mean:
                result.append(1)
            else:
                result.append(0)

        return result

    # 比较hash值
    def compHashCode(self, hc1, hc2):
        cnt = 0
        for i, j in zip(hc1, hc2):
            if i == j:
                cnt += 1
        return cnt

    # 计算感知哈希算法相似度
    def calpHashSimilarity(self):
        img1 = self.regularizeImage(self.img1)
        img2 = self.regularizeImage(self.img2)

        DCT1 = self.DCT(img1)
        DCT2 = self.DCT(img2)

        sub_list1 = self.submatrix_list(DCT1)
        sub_list2 = self.submatrix_list(DCT2)

        hc1 = self.getHashCode(sub_list1)
        hc2 = self.getHashCode(sub_list2)
        return self.compHashCode(hc1, hc2)


if __name__ == '__main__':
    file_search = "/Users/sunyihuan/Desktop/tt/65978163b01bdf8e.jpg"
    file_query_dir = "/Users/sunyihuan/Desktop/unlike"
    dir_list = os.listdir(file_query_dir)
    for i, file in enumerate(dir_list):
        if file != ".DS_Store":
            file_query = os.path.join(file_query_dir, file)
            pahsh_start = phash(file_query, file_search)
            print(pahsh_start.calpHashSimilarity())
