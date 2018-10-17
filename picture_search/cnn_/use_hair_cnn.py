# coding:utf-8 
'''
created on 2018/10/16

@author:sunyihuan
'''
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import scipy.io as scio
import time
from picture_search.utils import get_baseInfo_tx
from picture_search.picture_similar_move import picture_copy

hair_length_graph = tf.Graph()

model_path = "/Users/sunyihuan/Desktop/liuhai_model/model.ckpt-37"
# 98.34
MEAN = 0
STD = 1.0
IMG_SIZE = 96

with hair_length_graph.as_default():
    saver = tf.train.import_meta_graph("{}.meta".format(model_path))
    sess = tf.InteractiveSession(graph=hair_length_graph)
    saver.restore(sess, model_path)

    out = hair_length_graph.get_tensor_by_name("flatten/Reshape:0")
    features = hair_length_graph.get_tensor_by_name("features: 0")
    training_flag = hair_length_graph.get_tensor_by_name("is_training: 0")


def check_img_orientation(file_path):
    img = Image.open(file_path).convert("RGB")
    mirror = img
    if img.format == 'JPEG':
        info = img._getexif()
        if info:
            orientation = info.get(274, 0)
            if orientation == 1:
                mirror = img
            elif orientation == 2:
                # Vertical Mirror
                mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif orientation == 3:
                # Rotation 180°
                mirror = img.transpose(Image.ROTATE_180)
            elif orientation == 4:
                # Horizontal Mirror
                mirror = img.transpose(Image.FLIP_TOP_BOTTOM)
            elif orientation == 5:
                # Horizontal Mirror + Rotation 90° CCW
                mirror = img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_90)
            elif orientation == 6:
                # Rotation 270°
                mirror = img.transpose(Image.ROTATE_270)
            elif orientation == 7:
                # Horizontal Mirror + Rotation 270°
                mirror = img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.ROTATE_270)
            elif orientation == 8:
                # Rotation 90°
                mirror = img.transpose(Image.ROTATE_90)
            else:
                mirror = img
    return mirror


def get_region(points):
    '''
    获取裁剪区域（不拉伸）
    :param points:landmark坐标位置
    :return:包含目标区域
    '''
    x = points[:, 0]
    y = points[:, 1]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    wid = max((max_y - min_y), (max_x - min_x)) * 2

    region = ((max_x + min_x - wid) // 2, max_y - wid, (max_x + min_x + wid) // 2, max_y)
    return region


def corp_region(file_path, landmark_dict):
    '''
    裁剪图片的相应区域
    :param file_path:原始文件的全路径
    :return:Image对象
    '''
    if landmark_dict:
        face_profile = landmark_dict['face_profile']
        # 可能需要np对点进行合并 np.vstack() 或者np.concatenate()

        region = get_region(face_profile)
        img = check_img_orientation(file_path).convert("RGB").crop(region)
        img.save(file_path)
    else:
        img = check_img_orientation(file_path).convert("RGB")
    return img


def preprocess(trX):
    '''
    数据预处理
    :param trX: 输入的值
    :return: trX：处理后的值
    '''
    c = np.array(trX)
    # trX = (trX - c.mean()) / max(c.std(), 1.0 / IMG_SIZE)
    trX = c
    return trX


def predict(trX):
    '''
    :param trX: 输入特征
    :return: resY：预测值
    '''
    with hair_length_graph.as_default():
        resY = out.eval(session=sess, feed_dict={features: trX, training_flag: False})
        # resY = out.eval(session=sess, feed_dict={features: trX})
    return resY


def get_hair_length_bei(img_path):
    '''
    获取发长
    :param img_path: 图片地址
    :return:
    '''
    landmark_dict = get_baseInfo_tx(img_path)
    img = corp_region(img_path, landmark_dict)
    trX = preprocess(np.array(img.resize([IMG_SIZE, IMG_SIZE])).reshape([-1, IMG_SIZE, IMG_SIZE, 3]))

    res = predict(trX)
    return res


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
                file_res = get_hair_length_bei(os.path.join(file_dir, file))
                X.append(str(os.path.join(file_dir, file)))
                inceptionV3_value.append(file_res)
            except:
                print("error")
    scio.savemat("/Users/sunyihuan/Desktop/unlike.mat", {"X": X, "Y": inceptionV3_value})


def load_mat_data(mat_path):
    '''
    家在mat文件中的图片特征数据
    :return:
    '''
    data = scio.loadmat(mat_path)
    file_name = data["X"]
    file_inceptionData = data["Y"]
    return file_name, file_inceptionData


def picture_dist(p1, p2):
    '''
    计算两张图片的距离
    :param p1: 图片1的特征（卷积提取后的数据）
    :param p2: 图片2的特征（卷积提取后的数据）
    :return: 各点距离总和的平方
    '''
    aaa = np.array(p1) - p2
    aaa = aaa ** 2
    return sum(aaa[0])


def output_similar(file_search_path):
    '''
    输出最相似的图片名称
    :param file_search_path:
    :return:
    '''
    mat_path = "/Users/sunyihuan/Desktop/unlike.mat"

    res = np.array(get_hair_length_bei(file_search_path))
    file_name, file_inceptionData = load_mat_data(mat_path)
    dist = np.zeros((len(file_inceptionData), 1))
    for k in range(len(file_inceptionData)):
        dist[k] = picture_dist(file_inceptionData[k], res)
    fk = np.argmin(dist, 0)
    file_query = file_name[fk]
    file_query = str(file_query[0]).replace(" ", "")
    return file_query


if __name__ == "__main__":
    times = "firs0t"
    if times == "first":
        write_data_as_mat()

    save_root_dir = "/Users/sunyihuan/Desktop/tt/search"

    file_dir = "/Users/sunyihuan/Desktop/like"
    for file in os.listdir(file_dir):
        if file != ".DS_Store":
            try:
                file_search = os.path.join(file_dir, file)
                file_query = output_similar(file_search)
                picture_copy(save_root_dir, file_search, file_query)

            except:
                print("error:^**********")

    # mat_path = "/Users/sunyihuan/Desktop/unlike.mat"
    # file_name, file_inceptionData = load_mat_data(mat_path)
    # print(file_name[0])
    # print(len(file_name[0]))
