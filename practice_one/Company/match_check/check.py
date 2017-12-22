# coding:utf-8
'''
Created on 2017/12/19.

@author: chk01
'''
import numpy as np
import scipy.io as scio
import os
from aip import AipFace
import shutil

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

FaceShape = {
    'oval': "D",
    'round': "E",
    'square': "C",
    'triangle': "B",
    'heart': "D"
}
check_dir = 'C:/Users/chk01/Desktop/check'


# 本地图片
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def get_baseInfo(full_path):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark,faceshape,glasses"
    }
    res = client.detect(get_file_content(full_path), options=options)
    result = res['result'][0]
    landmark72 = result['landmark72']

    return landmark72


def compare_feature(org, feature):
    features = scio.loadmat('../load_material/feature_mat/{}'.format(org))
    target = features['data'][:] - feature
    top_index = np.argmin(np.linalg.norm(target, axis=(1, 2)))
    score = round(np.linalg.norm(target, axis=(1, 2))[top_index], 2)
    return top_index, score


def compare_face(faceshape, feature):
    typ = FaceShape[faceshape]
    features = scio.loadmat('feature_mat/face_{}'.format(typ))
    target = features['data'][:] - feature
    top_index = np.argmin(np.linalg.norm(target, axis=(1, 2)))
    print(faceshape, np.linalg.norm(target, axis=(1, 2))[top_index])
    return typ + '-' + str(top_index + 1)


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]
    return data


def point_to_vector(points):
    return points[1:] - points[:-1]


def read_feature(file_path):
    # step1 Api 获取脸型，五官点阵，是否有眼镜，脸型，性别
    landmark72 = get_baseInfo(file_path)

    # step2 数据预处理
    landmark72 = landmark72_trans(landmark72)

    left_eyebrow = point_to_vector(landmark72[22:30])
    left_eye = point_to_vector(landmark72[13:22])
    nose = point_to_vector(landmark72[47:58])
    lip = point_to_vector(landmark72[58:])

    right_eyebrow = point_to_vector(landmark72[39:47])
    right_eye = point_to_vector(landmark72[30:39])
    chin = landmark72[:13] - landmark72[6]

    return left_eye, right_eye, left_eyebrow, right_eyebrow, lip, nose, chin


def main():
    dir_path = os.listdir(check_dir)
    for file in [15]:
        file = str(file)
        print(file)
        face_dir = check_dir + '/' + file
        face_path = face_dir + '/face.jpg'
        left_eye, right_eye, left_eyebrow, right_eyebrow, lip, nose, chin = read_feature(face_path)

        left_eye_id, left_eye_score = compare_feature('left_eye', left_eye)
        right_eye_id, right_eye_score = compare_feature('right_eye', right_eye)
        left_eyebrow_id, left_eyebrow_score = compare_feature('left_eyebrow', left_eyebrow)
        right_eyebrow_id, right_eyebrow_score = compare_feature('right_eyebrow', right_eyebrow)
        lip_id, lip_score = compare_feature('lip', lip)
        nose_id, nose_score = compare_feature('nose', nose)

        feature_index = {
            'left_eye': left_eye_id,
            'right_eye': right_eye_id,
            'left_eyebrow': left_eyebrow_id,
            'right_eyebrow': right_eyebrow_id,
            'lip': lip_id,
            'nose': nose_id
        }
        for org in ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'lip', 'nose']:
            oldname = "../load_material/cartoon/{}/{}.png".format(org, feature_index[org] + 1)
            shutil.copyfile(oldname, face_dir + '/{}_{}.png'.format(org,feature_index[org] + 1))


if __name__ == '__main__':
    main()
