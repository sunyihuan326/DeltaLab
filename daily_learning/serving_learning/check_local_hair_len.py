# coding:utf-8 
'''
created on 2018/8/25

@author:sunyihuan
'''
from PIL import Image
import os
import tqdm
import shutil
import grpc
import time
from serving_learning.utils import get_baseInfo_tx
from serving_learning.hair_length_service import corp_region, get_baseInfo_tx, preprocess, get_region
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc, prediction_service_pb2


def check_img_orientation(file_path):
    img = Image.open(file_path)
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


def get_predict(im):
    server = "47.93.235.220:8501"

    channel = grpc.insecure_channel(server)

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = "inceptionV4_hair_len"
    request.model_spec.signature_name = 'predict_images'

    # put data into TensorProto and copy them into the request object

    request.inputs["images"].CopyFrom(
        tf.contrib.util.make_tensor_proto(im.astype(dtype=np.float32), shape=[1, 224, 224, 3]))

    result_future = stub.Predict.future(request, 10.0)
    result = result_future.result().outputs['scores'].int64_val

    return result[0]


def check_file(file_name, paste_path):
    '''
    检测本地图片的分类情况
    :param file_name: 待检测文件路径
    :return: res 模型预测
    '''
    if not os.path.exists(paste_path):  # 路径是否存在
        img1 = check_img_orientation(file_name)
        img1_size = img1.size
        w = int(max(img1_size[0], img1_size[1]) * 1.8)
        img2 = Image.new('RGB', (w, w), (255, 255, 255))

        left_min = max(0, int(w / 2) - int(img1_size[0] / 2))

        img2.paste(img1, box=(left_min, 0))
        img2.save(paste_path)

    landmark_dict = get_baseInfo_tx(paste_path)

    if landmark_dict['roll'] > 5 or landmark_dict['roll'] < -5:
        check_img_orientation(paste_path).rotate(-landmark_dict['roll']).save(paste_path)  # 图片旋转后保存

        landmark_dict = get_baseInfo_tx(paste_path)

    region_eyebrow = get_region(landmark_dict['face_profile'])

    im = check_img_orientation(paste_path).convert("RGB").crop(region_eyebrow)
    im = preprocess(np.array(im.resize([224, 224])).reshape([-1, 224, 224, 3]))
    res = get_predict(im)

    return res


def check_dir(file_dir, paste_dir):
    '''
    :param file_dir:待测试文件夹路径
    :param classes:int 分类数
    :return:
    '''
    file_list = os.listdir(file_dir)
    divide_dirs = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    for dir in divide_dirs:
        fdir = file_dir + '/' + dir
        if not os.path.exists(fdir):  # 路径是否存在
            os.makedirs(fdir)

    for file in file_list:
        if file != ".DS_Store" and file not in divide_dirs:
            try:
                print(file)
                file_path = os.path.join(file_dir, file)
                paste_path = os.path.join(paste_dir, file)
                start = time.time()
                model_res = check_file(file_path, paste_path)
                end = time.time()
                print(model_res, end - start)
                div_path = os.path.join(file_dir, divide_dirs[int(model_res)], file)
                shutil.move(file_path, div_path)
            except:
                print("error:", file)


if __name__ == '__main__':

    for k in ["zipai"]:
        file_dir = "/Users/sunyihuan/Desktop/Data/check_data/hair_length/0815fachang-yan_check0_1/{}".format(k)
        paste_dir = "/Users/sunyihuan/Desktop/Data/check_data/hair_length/0815fachang-yan_check0_1/{}_pasted".format(k)
        if not os.path.exists(paste_dir):  # 路径是否存在
            os.makedirs(paste_dir)

        check_dir(file_dir, paste_dir)
