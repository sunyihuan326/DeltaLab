# coding:utf-8
'''
Created on 2017/11/27.

@author: chk01
'''
from aip import AipFace
import urllib.request
import numpy as np
import scipy.io as scio
from PIL import Image

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
CartoonPoint = scio.loadmat("CartoonPoint")
Glasses = [300, 470]


# 本地图片
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


# 地址图片
def get_url_img(filePath):
    image_bytes = urllib.request.urlopen(filePath).read()
    return image_bytes


def get_baseInfo(full_path):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark,faceshape,gender,glasses,qualities"
    }
    res = client.detect(get_file_content(full_path), options=options)
    assert res['result_num'] == 1
    result = res['result'][0]
    assert result['face_probability'] > 0.8

    isPerson = result['qualities']['type']['human']
    isCartoon = result['qualities']['type']['cartoon']
    assert isPerson > isCartoon

    landmark72 = result['landmark72']
    gender = result['gender']
    glasses = result['glasses']
    faceshape = sorted(result['faceshape'], key=lambda x: -x['probability'])
    # oval,round,square,triangle,heart
    # print(faceshape)
    return landmark72, gender, glasses, faceshape[0]['type']


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]
    return data


def point_to_vector(points):
    return points[1:] - points[:-1]


def organ_struct(landmark72):
    # reb,reye,nose,lip,chin,leb,leye
    return [(landmark72[24] + landmark72[28]) / 2, landmark72[21],
            landmark72[57], landmark72[70],
            landmark72[6], (landmark72[41] + landmark72[45]) / 2,
            landmark72[38]]


def org_alignment(org_struct):
    eb_y = (org_struct[0][1] + org_struct[-2][1]) / 2
    eye_y = (org_struct[1][1] + org_struct[-1][1]) / 2
    eye_dis = (org_struct[-1][0] - org_struct[1][0]) / 2
    eb_dis = (org_struct[-2][0] - org_struct[0][0]) / 2
    org_struct[0][1] = eb_y
    org_struct[-2][1] = eb_y
    org_struct[1][1] = eye_y
    org_struct[-1][1] = eye_y
    center_x = (org_struct[2][0] + org_struct[3][0] + org_struct[4][0]) / 3
    org_struct[2][0] = center_x
    org_struct[3][0] = center_x
    org_struct[4][0] = center_x

    org_struct[1][0] = center_x - eye_dis
    org_struct[-1][0] = center_x + eye_dis

    org_struct[0][0] = center_x - eb_dis
    org_struct[-2][0] = center_x + eb_dis

    return org_struct


def read_feature(file_path):
    # step1 Api 获取脸型，五官点阵，是否有眼镜，脸型，性别
    landmark72, gender, glasses, faceshape = get_baseInfo(file_path)

    # step2 数据预处理
    landmark72 = landmark72_trans(landmark72)

    left_eyebrow = point_to_vector(landmark72[22:30])
    left_eye = point_to_vector(landmark72[13:22])
    nose = point_to_vector(landmark72[47:58])
    lip = point_to_vector(landmark72[58:])
    width = landmark72[12][0] - landmark72[0][0]
    right_eyebrow = point_to_vector(landmark72[39:47])
    right_eye = point_to_vector(landmark72[30:39])
    chin = landmark72[:13] - landmark72[6]
    org_struct = org_alignment(organ_struct(landmark72))
    return left_eye, right_eye, left_eyebrow, right_eyebrow, lip, nose, chin, org_struct, width, glasses, faceshape


def compare_feature(org, feature):
    features = scio.loadmat('feature_mat/{}'.format(org))
    target = features['data'][:] - feature
    top_index = np.argmin(np.linalg.norm(target, axis=(1, 2)))
    print(org, np.linalg.norm(target, axis=(1, 2))[top_index])
    return top_index


def compare_face(faceshape, feature):
    typ = FaceShape[faceshape]
    features = scio.loadmat('feature_mat/face_{}'.format(typ))
    target = features['data'][:] - feature
    top_index = np.argmin(np.linalg.norm(target, axis=(1, 2)))
    print('11', faceshape, np.linalg.norm(target, axis=(1, 2))[top_index])
    return typ + '-' + str(top_index + 1)


def get_carton_points(feature_index):
    cartoon_points = []
    # 顺序固定
    for org in ['left_eyebrow', 'left_eye', 'nose', 'lip', 'chin', 'right_eyebrow', 'right_eye']:
        if org == 'chin':
            typ, _id = feature_index[org].split('-')
            data = CartoonPoint[typ + '_shape'][int(_id) - 1][2:]
        else:
            data = CartoonPoint[org][feature_index[org]]
        cartoon_points.append(data)
    return cartoon_points


def merge_all(real_width, real_points, feature_index, face_id):
    cartoon_points = get_carton_points(feature_index)
    image = Image.open('cartoon/face/{}.png'.format(face_id)).convert('RGBA')

    typ, fid = face_id.split('-')
    face_data = CartoonPoint[typ + '_shape'][int(fid) - 1]
    ratio_x = face_data[0] / real_width
    # print(ratio_x)
    ear_height = face_data[1]
    chin_point = cartoon_points[4]
    real_chin_point = real_points[4]

    ratio_y = (ear_height - chin_point[1]) / (np.array(real_points) - real_chin_point)[1][1]
    # print(ratio_y)

    norm_real_points = (np.array(real_points) - real_chin_point) * [ratio_x, ratio_y]

    boxes = norm_real_points - cartoon_points + chin_point

    # 调整位置和耳朵平齐
    # eye_height = boxes[1][1]
    # cartoon_eye = cartoon_points[1][1] + eye_height
    # print(ear_height - cartoon_eye)
    # boxes = boxes + [0, ear_height - cartoon_eye]
    # print('boxes===', boxes)
    TypOrgans = ['left_eyebrow', 'left_eye', 'nose', 'lip', 'chin', 'right_eyebrow', 'right_eye']

    norm_real_glasses = (norm_real_points[1] + norm_real_points[-1]) / 2
    glasses_box = norm_real_glasses + [0, 35] + chin_point - Glasses

    for i, org in enumerate(TypOrgans):
        if org != 'chin':
            organ = Image.open("cartoon/{}/{}.png".format(org, feature_index[org] + 1))
            image.paste(organ, list(boxes[i].astype(np.int)), mask=organ)
    if feature_index['glasses'] == 1:
        organ = Image.open("cartoon/{}/{}_{}.png".format('glasses', 'glasses', 1))
        image.paste(organ, list(np.squeeze(glasses_box).astype(np.int)), mask=organ)
    return image


def main(file_path):
    # step1 获取所有特征数据
    left_eye, _ry, left_eyebrow, _rb, lip, nose, chin, org_struct, width, glasses, faceshape = read_feature(file_path)

    # step2 获取Cartoon匹配序号
    left_eye_id = compare_feature('left_eye', left_eye)
    # right_eye_id = compare_feature('right_eye', right_eye)
    left_eyebrow_id = compare_feature('left_eyebrow', left_eyebrow)
    # right_eyebrow_id = compare_feature('right_eyebrow', right_eyebrow)
    lip_id = compare_feature('lip', lip)
    nose_id = compare_feature('nose', nose)
    face_id = compare_face(faceshape, chin)

    print(face_id)
    feature_index = {
        'left_eye': left_eye_id,
        'right_eye': left_eye_id,
        'left_eyebrow': left_eyebrow_id,
        'right_eyebrow': left_eyebrow_id,
        'lip': lip_id,  # np.random.randint(0, 20, 1)[0]
        'nose': nose_id,
        'glasses': glasses,
        'chin': face_id
    }
    print(feature_index)

    image = merge_all(width, org_struct, feature_index, face_id)
    image.save('res' + file_path.replace('jpg', 'png'))
    image.show()
    return True


if __name__ == "__main__":
    file_path = 'D1.jpg'
    main(file_path)
