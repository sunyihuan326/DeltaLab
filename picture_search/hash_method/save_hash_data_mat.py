# coding:utf-8 
'''
created on 2018/10/23

@author:sunyihuan
'''
import os
import scipy.io as scio
import numpy as np
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
                # img = img_process(os.path.join(file_query_dir, file))
                img = np.array(img_process(os.path.join(file_query_dir, file)).convert("L"))
                hash0 = imghash()
                file_res = hash0.getHashCode(img)
                X.append(str(os.path.join(file_query_dir, file)))
                inceptionV3_value.append(file_res)
            except:
                print("error")
    scio.savemat("/Users/sunyihuan/Desktop/unlike_{}.mat".format(hash), {"X": X, "Y": inceptionV3_value})


if __name__ == "__main__":
    file_query_dir = "/Users/sunyihuan/Desktop/unlike"
    write_data_as_mat(file_query_dir, hash="imghash")
