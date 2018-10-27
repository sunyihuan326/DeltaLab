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
from picture_search.picture_similar_move import picture_copy
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


def write_data_as_mat(file_query_dir, hash="ahash"):
    '''
    将背搜索的图片特征提取后保存为mat格式
    :return:
    '''
    X = []
    inceptionV3_value = []
    for i, file in enumerate(os.listdir(file_query_dir)):
        if file != ".DS_Store":
            try:
                print(file)
                img = np.array(img_process(os.path.join(file_query_dir, file)).convert("L"))
                file_res = hash0.getHashCode(img)
                X.append(str(os.path.join(file_query_dir, file)))
                inceptionV3_value.append(file_res)
            except:
                print("error")
    scio.savemat("/Users/sunyihuan/Desktop/kongqi/1_{}.mat".format(hash), {"X": X, "Y": inceptionV3_value})


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

    mat_path = "/Users/sunyihuan/Desktop/kongqi/1_{}.mat".format(hash)

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
    file_query_dir = "/Users/sunyihuan/Desktop/kongqi/kongqi1"

    hash0 = imghash()  # 选择hash方法
    mat_hash = "imghash"

    times = "first"
    if times == "first":
        write_data_as_mat(file_query_dir, mat_hash)

    file_search_dir = "/Users/sunyihuan/Desktop/like1"  # 目标图文件夹

    save_root_dir = "/Users/sunyihuan/Desktop/kongqi/{}_kongqi1".format(mat_hash)  # 要保存的文件夹

    for file in os.listdir(file_search_dir):
        if file != ".DS_Store":
            try:
                file_search = os.path.join(file_search_dir, file)
                file_query = output_similar(file_search, mat_hash)
                picture_copy(save_root_dir, file_search, file_query)

            except:
                print("error:^**********")
    b = time.time()
    print("time:", b - a)
