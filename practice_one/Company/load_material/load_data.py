# coding:utf-8
'''
Created on 2017/11/24.

@author: chk01
'''
from skimage import feature as ft
from PIL import Image
from aip import AipFace
import scipy.io as scio
import os
import urllib.request
import numpy as np

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '

client = AipFace(APP_ID, API_KEY, SECRET_KEY)
root_dir = 'C:/Users/chk01/Desktop/Delta/image'
TypOrgans = ['face', 'lip', 'nose', 'left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow']
PointNum = {
    'face': 13,
    'lip': 14,
    'nose': 11,
    'left_eye': 9,
    'right_eye': 9,
    'left_eyebrow': 8,
    'right_eyebrow': 8
}


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def get_url_img(filePath):
    image_bytes = urllib.request.urlopen(filePath).read()
    return image_bytes


# nose eye
def get_hog_feature():
    feature = {
        'left_eye': '',
        'right_eye': '',
        'nose': '',
    }
    for org in ['nose', 'left_eye', 'right_eye']:
        organ = Image.open("real/{}/{}_{}.jpg".format(org, org, 2)).convert("L")
        feature[org] = ft.hog(organ,  # input image
                              orientations=9,  # number of bins
                              pixels_per_cell=(4, 4),  # pixel per cell
                              cells_per_block=(10, 10),  # cells per blcok
                              block_norm='L2-Hys',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                              transform_sqrt=False,  # power law compression (also known as gamma correction)
                              feature_vector=True,  # flatten the final vectors
                              visualise=True)  # return HOG map
        print(feature[org][0])
        res = Image.fromarray(feature[org][1] * 200)
        res.show()
    return feature


def point_to_vector(points):
    data = np.zeros([len(points), 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]
    return data[1:] - data[:-1]


def get_org_point(landmark72, org):
    if org == 'face':
        data = np.zeros([len(landmark72[:13]), 2])
        data[:, 0] = [p['x'] for p in landmark72[:13]]
        data[:, 1] = [p['y'] for p in landmark72[:13]]
        data = data - data[6]
    elif org == 'left_eye':
        data = np.array(point_to_vector(landmark72[13:22]))
    elif org == 'right_eye':
        data = np.array(point_to_vector(landmark72[30:39]))
    elif org == 'left_eyebrow':
        data = np.array(point_to_vector(landmark72[22:30]))
    elif org == 'right_eyebrow':
        data = np.array(point_to_vector(landmark72[39:47]))
    elif org == 'lip':
        data = np.array(point_to_vector(landmark72[58:]))
    else:
        data = np.array(point_to_vector(landmark72[47:58]))

    return data


def get_landmark72(full_path):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark"
    }
    result = client.detect(get_file_content(full_path), options=options)
    landmark72 = result['result'][0]['landmark72']
    return landmark72


# chin 13# data1 = data[:13]
# eyes# data2 = data[13:22]# data2.extend(data[30:39])
# browns# data3 = data[22:30]# data3.extend(data[39:47])
# nose# data4 = data[47:58]
# mouse# data5 = data[58:]
# other
def get_point_feature():
    for org in ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'nose', 'lip']:
        print('开始{}导入'.format(org))
        dir_path = os.listdir(root_dir + '/src/' + org)
        m = len(dir_path)
        n = PointNum[org] - 1
        data = np.zeros([m, n, 2])
        for i, sourceDir in enumerate(dir_path):
            _id = int(sourceDir.split('.')[0]) - 1
            full_path = root_dir + '/src/' + org + '/' + sourceDir
            landmark72 = get_landmark72(full_path)
            _data = get_org_point(landmark72, org)
            data[_id] = _data
            print('load--->{}---图{}'.format(org, _id))
        scio.savemat('feature_mat/' + org, {"data": data})
        print('完成{}导入'.format(org))


def get_face_feature():
    for typ in ['A', 'B', 'C', 'D', 'E']:
        print('开始{}型导入'.format(typ))
        dir_path = os.listdir(root_dir + '/src/face_' + typ)
        m = len(dir_path)
        n = 13
        X = np.zeros([m, n, 2])
        Y = np.zeros([m, ])
        for i, sourceDir in enumerate(dir_path):
            _id = int(sourceDir.split('.')[0]) - 1
            full_path = root_dir + '/src/face_' + typ + '/' + sourceDir
            landmark72 = get_landmark72(full_path)
            _data = get_org_point(landmark72, 'face')
            X[_id] = _data
            Y[_id] = _id + 1
            print('load--->{}---图{}'.format(typ, _id))
        scio.savemat('feature_matrix/face_' + typ, {"X": X, "Y": Y})
        print('完成{}导入'.format(typ))


if __name__ == '__main__':
    get_face_feature()
    # get_point_feature()
    # data = scio.loadmat('feature_mat/left_eye')
    # print(data['data'].shape)
