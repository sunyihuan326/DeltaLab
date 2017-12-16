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
FaceId = {
    'oval': 2,
    'round': 2,
    'square': 3,
    'triangle': 1,
    'heart': 0
}
FaceData = [[383, 500], [383, 500], [383, 500]]
# 最终需要文件导入
CartoonPoint = {
    'left_eye': [{'mp': [50, 37]}]*20,
    'right_eye': [{'mp': [75, 37]}]*20,
    'left_eyebrow': [{'mp': [50, 27]}]*20,
    'right_eyebrow': [{'mp': [100, 27]}]*20,
    'chin': [{'mp': [191, 478]}]*20,
    'nose': [{'mp': [50, 117]}]*20,
    'lip': [{'mp': [67, 42]}]*20,
    'glasses': [300, 470]
}


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

    org_struct = org_alignment(organ_struct(landmark72))
    return left_eye, right_eye, left_eyebrow, right_eyebrow, lip, nose, org_struct, width, glasses, faceshape


def compare_feature(org, feature):
    features = scio.loadmat('feature_mat/{}'.format(org))
    m, _, _ = features['data'].shape
    target = features['data'][:] - feature
    top_index = np.argmin(np.linalg.norm(target, axis=(1, 2)))
    return top_index


def get_carton_points(feature_index):
    cartoon_points = []
    # 顺序固定
    for org in ['left_eyebrow', 'left_eye', 'nose', 'lip', 'chin', 'right_eyebrow', 'right_eye']:
        cartoon_points.append(CartoonPoint[org][feature_index[org]]['mp'])
    return cartoon_points


def merge_all(real_width, real_points, feature_index, face_id):
    cartoon_points = get_carton_points(feature_index)
    image = Image.open('cartoon/face/face_{}.png'.format(face_id + 1)).convert('RGBA')
    ratio = FaceData[face_id][0] / real_width
    # print('ratio===', ratio)
    lip_point = [FaceData[face_id][0] // 2, FaceData[face_id][1]]
    # print('lip_point===', lip_point)
    real_lip_point = real_points[4]
    # print('real_lip_point===', real_lip_point)
    norm_real_points = (np.array(real_points) - real_lip_point) * ratio + real_lip_point
    # print('norm_real_points==', norm_real_points)
    boxes = norm_real_points - cartoon_points + lip_point - real_lip_point
    # print('boxes===', boxes)
    TypOrgans = ['left_eyebrow', 'left_eye', 'nose', 'lip', 'chin', 'right_eyebrow', 'right_eye']

    norm_real_glasser = (norm_real_points[1] + norm_real_points[-1]) / 2
    glasses_box = norm_real_glasser + [0, 35] + lip_point - real_lip_point - CartoonPoint['glasses']

    for i, org in enumerate(TypOrgans):
        if org != 'chin':
            organ = Image.open("cartoon/{}/{}.png".format(org, feature_index[org] + 1))
            image.paste(organ, list(boxes[i].astype(np.int)), mask=organ)
    if feature_index['glasses'] == 1:
        organ = Image.open("cartoon/{}/{}_{}.png".format('glasses', 'glasses', 1))
        image.paste(organ, list(glasses_box.astype(np.int)), mask=organ)
    return image


def main(file_path):
    # step1 获取所有特征数据
    left_eye, _ry, left_eyebrow, _rb, lip, nose, org_struct, width, glasses, faceshape = read_feature(file_path)

    # step2 获取Cartoon匹配序号
    left_eye_id = compare_feature('left_eye', left_eye)
    # right_eye_id = compare_feature('right_eye', right_eye)
    left_eyebrow_id = compare_feature('left_eyebrow', left_eyebrow)
    # right_eyebrow_id = compare_feature('right_eyebrow', right_eyebrow)
    lip_id = compare_feature('lip', lip)
    nose_id = compare_feature('nose', nose)
    face_id = FaceId[faceshape]
    # face_id=1
    feature_index = {
        'left_eye': left_eye_id,
        'right_eye': left_eye_id,
        'left_eyebrow': left_eyebrow_id,
        'right_eyebrow': left_eyebrow_id,
        'lip': lip_id,
        'nose': nose_id,
        'glasses': glasses,
        'chin': face_id
    }
    print(feature_index)
    # assert 1==0
    image = merge_all(width, org_struct, feature_index, face_id)
    image.save('res' + file_path.replace('jpg', 'png'))
    image.show()
    return True


if __name__ == "__main__":
    # import tqdm
    # for i in tqdm(range(10), total=1, ncols=1, leave=False, unit='b'):
    #     print(i)
    file_path = '2.jpg'
    main(file_path)
