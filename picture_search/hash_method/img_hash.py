# coding:utf-8 
'''
created on 2018/10/16

@author:sunyihuan
'''
import os
import scipy.io as scio
import numpy as np
import time
from PIL import Image
import hashlib
import struct


def remove_transparent_pixels(img):
    pixels = img.getdata()
    new_data = []
    for item in pixels:
        if item[3] == 0:
            new_data.append((0, 0, 0, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)


def get_hash(img):
    img = img.convert('RGBA')
    remove_transparent_pixels(img)
    data = img.tobytes()

    h = hashlib.sha512()
    size_prefix = struct.pack('!LL', *img.size)
    h.update(size_prefix)
    h.update(data)
    return h


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


def write_data_as_mat():
    '''
    将背搜索的图片特征提取后保存为mat格式
    :return:
    '''
    file_dir = "/Users/sunyihuan/Desktop/unlike"
    X = []
    inceptionV3_value = []
    for i, file in enumerate(os.listdir(file_dir)):
        if file != ".DS_Store":
            try:
                print(file)
                file_res = get_hash(os.path.join(file_dir, file)).hexdigest()
                X.append(str(os.path.join(file_dir, file)))
                inceptionV3_value.append(file_res)
            except:
                print("error")
    scio.savemat("/Users/sunyihuan/Desktop/unlike_hash.mat", {"X": X, "Y": inceptionV3_value})


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
        if a[i] != b[i]:
            dist = dist + 1
    return dist


def output_similar(file_search_path):
    '''
    输出最相似的图片名称
    :param file_search_path:
    :return:
    '''
    mat_path = "/Users/sunyihuan/Desktop/unlike_hash.mat"

    img = img_process(file_search_path)
    res = get_hash(img).hexdigest()
    file_name, file_inceptionData = load_mat_data(mat_path)
    dist = np.zeros((len(file_inceptionData), 1))
    for k in range(len(file_inceptionData)):
        dist[k] = getMH(file_inceptionData[k], res)
    fk = np.argmin(dist, 0)
    file_query = file_name[fk]
    file_query = str(file_query[0]).replace(" ", "")
    return file_query


if __name__ == "__main__":
    times = "first0"
    if times == "first":
        write_data_as_mat()
    a = time.time()

    file_search = "/Users/sunyihuan/Desktop/tt/65978163b01bdf8e.jpg"
    image = img_process(file_search)
    img_data = get_hash(image).hexdigest()  # hash值展示
    print(len(img_data))

    file_query = output_similar(file_search)
    b = time.time()
    # print("time:", b - a)
    print(file_query)
    Image.open(file_query).show()

    # 094613cb34a3a8b709493caaa719736ec10ae5a88187e574ce696d9dc197b557
