# coding:utf-8 
'''
created on 2018/10/23

@author:sunyihuan
'''
import os
import scipy.io as scio
import numpy as np
import time
from PIL import Image
from picture_search.hash_method.ahash import ahash  # ahash
from picture_search.hash_method.dhash import dhash  # dhash
from picture_search.hash_method.phash import phash  # phash
from picture_search.hash_method.imghash import imghash  # 图片直接hash


def padding_(file_path):
    '''
    图片白色填充为正方形
    :param file_path: 图片地址
    :return: 填充后的图片
    '''
    img = Image.open(file_path).convert("RGB")
    a = img.size
    if a[0] != a[1]:
        pic_w = max(a[0], a[1])
        img = np.array(img)
        channel_one = img[:, :, 0]
        channel_two = img[:, :, 1]
        channel_three = img[:, :, 2]
        padding_x = int((pic_w - a[1]) / 2)
        padding_y = int((pic_w - a[0]) / 2)
        channel_one = np.pad(channel_one, ((padding_x, padding_x), (padding_y, padding_y)), 'constant',
                             constant_values=(255, 255))
        channel_two = np.pad(channel_two, ((padding_x, padding_x), (padding_y, padding_y)), 'constant',
                             constant_values=(255, 255))
        channel_three = np.pad(channel_three, ((padding_x, padding_x), (padding_y, padding_y)), 'constant',
                               constant_values=(255, 255))
        image = np.dstack((channel_one, channel_two, channel_three))
        image = Image.fromarray(image)
    else:
        image = img

    return image


def img_process(file_search):
    image = padding_(file_search)
    return image.resize((64, 64))


def load_mat_data(mat_path):
    '''
    家在mat文件中的图片特征数据
    :return:
    '''
    data = scio.loadmat(mat_path)
    file_name = data["X"]
    file_inceptionData = data["Y"]
    return file_name, file_inceptionData


def getMH(a, b):  # 比较100个字符有几个字符相同
    dist = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            dist = dist + 1
    return dist


def output_similar(file_search_path, hash="ahash"):
    '''
    输出最相似的图片名称
    :param file_search_path:目标图片
    :param hash：hash方法
    :return:
    '''

    mat_path = "/Users/sunyihuan/Desktop/unlike_{}.mat".format(hash)

    img = img_process(file_search_path)

    res = hash0.getHashCode(np.array(img.convert("L")))
    file_name, file_inceptionData = load_mat_data(mat_path)
    dist = np.zeros((len(file_inceptionData), 1))
    for k in range(len(file_inceptionData)):
        dist[k] = getMH(file_inceptionData[k], res)
    fk = np.argmax(dist)
    file_query = file_name[fk]
    file_query = str(file_query).replace(" ", "")
    return file_query


if __name__ == "__main__":
    a = time.time()
    file_search = "/Users/sunyihuan/Desktop/tt/65978163b01bdf8e.jpg"

    hash0 = ahash()  # 选择hash方法
    mat_hash = "ahash"

    file_query = output_similar(file_search, hash=mat_hash)
    b = time.time()
    print("time:", b - a)
    print(file_query)
    Image.open(file_query).show()
