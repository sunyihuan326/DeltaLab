# coding:utf-8 
'''
created on 2018/10/22

@author:sunyihuan
'''

from PIL import Image
import numpy as np
import os


class ahash(object):
    def __init__(self, img1_path, img2_path):
        self.img1 = Image.open(img1_path)
        self.img2 = Image.open(img2_path)

    # 正则化图像
    def regularizeImage(self, img, size=(256, 256)):
        return np.array(img.resize(size).convert('L'))

    # 计算hash值
    def getHashCode(self, img, size=(8, 8)):
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

    # 比较hash值
    def compHashCode(self, hc1, hc2):
        cnt = 0
        for i, j in zip(hc1, hc2):
            if i == j:
                cnt += 1
        return cnt

    # 计算平均哈希算法相似度
    def calaHashSimilarity(self):
        img1 = self.regularizeImage(self.img1)
        img2 = self.regularizeImage(self.img2)
        hc1 = self.getHashCode(img1)
        hc2 = self.getHashCode(img2)
        return self.compHashCode(hc1, hc2)


if __name__ == "__main__":
    file_search = "/Users/sunyihuan/Desktop/tt/65978163b01bdf8e.jpg"
    file_query_dir = "/Users/sunyihuan/Desktop/unlike"
    dir_list = os.listdir(file_query_dir)
    for i, file in enumerate(dir_list):
        if file != ".DS_Store":
            file_query = os.path.join(file_query_dir, file)
            ahash0 = ahash(file_search, file_query)
            print(ahash0.calaHashSimilarity())
