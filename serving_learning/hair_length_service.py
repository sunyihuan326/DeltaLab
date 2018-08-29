# coding:utf-8 
'''
created on 2018/6/26

@author:sunyihuan
'''
import tensorflow as tf
import numpy as np
from PIL import Image
from serving_learning.utils import get_baseInfo_tx

hair_length_graph = tf.Graph()

# model_path = "/Users/sunyihuan/Desktop/parameters/hair_length/second/complex75.03_simple78.63/model.ckpt-23"  # check4
model_path = "/Users/sunyihuan/Desktop/parameters/hair_length/second/complex77_simple70/model.ckpt-39"  # check0
# 98.34
MEAN = 0
STD = 1.0
IMG_SIZE = 224

with hair_length_graph.as_default():
    saver = tf.train.import_meta_graph("{}.meta".format(model_path))
    sess = tf.InteractiveSession(graph=hair_length_graph)
    saver.restore(sess, model_path)

    predict_op = hair_length_graph.get_tensor_by_name("classes:0")
    features = hair_length_graph.get_tensor_by_name("features: 0")
    # training_flag = hair_length_graph.get_tensor_by_name("Placeholder:0")


def preprocess(trX):
    '''
    数据预处理
    :param trX: 输入的值
    :return: trX：处理后的值
    '''
    c = np.array(trX)
    trX = (trX - c.mean()) / max(c.std(), 1.0 / IMG_SIZE)
    # trX = (trX - MEAN) / STD
    # trX[:, :, :, 0] = (c[:, :, :, 0] - np.mean(c[:, :, :, 0])) / np.std(c[:, :, :, 0])
    # trX[:, :, :, 1] = (c[:, :, :, 1] - np.mean(c[:, :, :, 1])) / np.std(c[:, :, :, 1])
    # trX[:, :, :, 2] = (c[:, :, :, 2] - np.mean(c[:, :, :, 2])) / np.std(c[:, :, :, 2])
    return trX


def get_region(points):
    '''
    获取裁剪区域（不拉伸）
    :param points:landmark坐标位置
    :return:包含目标区域
    '''
    d = points[10][1] - points[0][1]

    xmin = max(0, int(points[10][0] - 2.7 * d))
    ymin = max(0, int(points[0][1] - 1.4 * d))

    region = (xmin, ymin, xmin + 5.4 * d, ymin + 5.4 * d)
    return region


def corp_region(file_path, landmark_dict):
    '''
    裁剪图片的相应区域
    :param file_path:原始文件的全路径
    :return:Image对象
    '''
    face_profile = landmark_dict['face_profile']
    # 可能需要np对点进行合并 np.vstack() 或者np.concatenate()

    region = get_region(face_profile)
    # Image.open(file_path).save("source.jpg")
    img = Image.open(file_path).convert("RGB")
    # img = Image.open(file_path)
    # img = check_img_orientation(file_path).convert("RGB").crop(region)

    img_size = img.size
    w = int(max(img_size[0], img_size[1]) * 1.8)
    img_new = Image.new('RGB', (w, w), (255, 255, 255))

    left_min = max(0, int(w / 2) - int(img_size[0] / 2))
    img_new.paste(img, box=(left_min, 0))
    # img_new.save('paste.png')

    img_new = img_new.crop((region[0] + left_min, region[1], region[2] + left_min, region[3]))
    # img_new.save('paste_crop.png')
    # img.save('res-len.jpg')
    return img_new


def predict(trX):
    '''
    :param trX: 输入特征
    :return: resY：预测值
    '''
    with hair_length_graph.as_default():
        # resY = predict_op.eval(session=sess, feed_dict={features: trX, training_flag: False})
        resY = predict_op.eval(session=sess, feed_dict={features: trX})
    return resY


def get_hair_length(img_path, landmark_dict):
    '''
    获取发长
    :param img_path: 图片地址
    :param landmark_dict: 腾讯返回的72个点
    :return:
    '''
    img = corp_region(img_path, landmark_dict)

    trX = preprocess(np.array(img.resize([IMG_SIZE, IMG_SIZE])).reshape([-1, IMG_SIZE, IMG_SIZE, 3]))

    res = np.squeeze(predict(trX))
    return res

#
# image_path = "/Users/sunyihuan/Desktop/Data/check_data/hair_length/0815fachang-yan_check4/tapai/1/10667.jpg"
#
# landmark_dict = get_baseInfo_tx(image_path)
# print(get_hair_length(image_path, landmark_dict))
