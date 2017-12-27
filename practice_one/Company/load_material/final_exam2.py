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
import os
import math

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '
root_dir = 'C:/Users/chk01/Desktop/Delta/image'
client = AipFace(APP_ID, API_KEY, SECRET_KEY)
FaceShape = {
    'oval': "C",
    'round': "E",
    'square': "C",
    'triangle': "B",
    'heart': "C"
}
CartoonPoint = scio.loadmat("CartoonPoint")
Glasses = [164, 65]


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
    angle = result['rotation_angle']

    return landmark72, angle, gender, glasses, faceshape[0]['type']


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]
    return data


def point_to_vector(points):
    return points[1:] - points[:-1]


def point2feature_ebr(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    center = (points[0] + points[4]) / 2
    feature = (points - center) / np.array([wid, hei])
    # feature = (points - center)
    return feature


def point2feature_eye(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    center = points[-1]
    feature = (points - center) / np.array([wid, hei])
    # feature = (points - center)
    return feature


def point2feature_nose(landmarks):
    points = landmarks[49:55]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    center = landmarks[57]
    feature = (points - center) / np.array([wid, hei])
    # feature = (points - center)
    return feature


def point2feature_lip(landmarks):
    point1 = [landmarks[58], landmarks[59], landmarks[60], landmarks[61],
              landmarks[68], landmarks[67], landmarks[66]]
    x1 = [p[0] for p in point1]
    y1 = [p[1] for p in point1]
    wid1 = max(x1) - min(x1)
    hei1 = max(y1) - min(y1)

    center1 = landmarks[67]
    feature1 = (point1 - center1) / np.array([wid1, hei1])
    # feature1 = (point1 - center1)

    point2 = [landmarks[58], landmarks[65], landmarks[64], landmarks[63],
              landmarks[71], landmarks[70], landmarks[69]]
    x2 = [p[0] for p in point2]
    y2 = [p[1] for p in point2]
    wid2 = max(x2) - min(x2)
    hei2 = max(y2) - min(y2)

    center2 = landmarks[70]
    feature2 = (point2 - center2) / np.array([wid2, hei2])
    # feature2 = (point2 - center2)
    feature = np.zeros([14, 2])
    feature[:7, :] = feature1
    feature[7:, :] = feature2

    return feature


def organ_struct(landmark72):
    # leb,leye,nose,lip,chin,reb,reye
    return [(landmark72[24] + landmark72[28]) / 2, landmark72[21],
            landmark72[57], (landmark72[70] + landmark72[67]) / 2,
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


def get_box(points):
    x = points[:, 0]
    y = points[:, 1]
    return max(x) - min(x), max(y) - min(y)


def get_relative_position(points):
    wid, hei = get_box(points[0:13])
    chin = points[6]

    left_eyebr_point = points[22:30]
    left_eyebr_wid, left_eyebr_hei = get_box(left_eyebr_point)
    left_eyebr = [(points[24] + points[28]) / 2, left_eyebr_wid / wid, left_eyebr_hei / hei]

    right_eyebr_point = points[39:47]
    right_eyebr_wid, right_eyebr_hei = get_box(right_eyebr_point)
    right_eyebr = [(points[41] + points[45]) / 2, right_eyebr_wid / wid, right_eyebr_hei / hei]

    eyebrs_width_ratio = (right_eyebr[0][0] - left_eyebr[0][0]) / wid
    eyebrs_height_ratio = np.abs((right_eyebr[0][1] + left_eyebr[0][1]) / 2 - chin[1]) / hei

    left_eye_point = points[13:22]
    left_eye_wid, left_eye_hei = get_box(left_eye_point)
    left_eye = [points[21], left_eye_wid / wid, left_eye_hei / hei]

    right_eye_point = points[30:39]
    right_eye_wid, right_eye_hei = get_box(right_eye_point)
    right_eye = [points[38], right_eye_wid / wid, right_eye_hei / hei]

    eyes_width_ratio = (right_eye[0][0] - left_eye[0][0]) / wid
    eyes_height_ratio = np.abs((right_eye[0][1] + left_eye[0][1]) / 2 - chin[1]) / hei

    nose_point = points[49:55]
    nose_wid, nose_hei = get_box(nose_point)
    nose = [points[57], nose_wid / wid, nose_hei / hei]
    nose_height_ratio = np.abs(nose[0][1] - chin[1]) / hei

    lip_point = points[58:72]
    lip_wid, lip_hei = get_box(lip_point)
    lip = [(points[67] + points[70]) / 2, lip_wid / wid, lip_hei / hei]
    lip_height_ratio = np.abs(lip[0][1] - chin[1]) / hei

    organ_boxes = {
        'eye_br': [eyebrs_width_ratio, eyebrs_height_ratio, (left_eyebr[1] + right_eyebr[1]) / 2,
                   (left_eyebr[2] + right_eyebr[2]) / 2],  # [眉间距比例，眉高比例，眉毛宽比，眉毛高比]
        'eye': [eyes_width_ratio, eyes_height_ratio, (left_eye[1] + right_eye[1]) / 2,
                (left_eye[2] + right_eye[2]) / 2],  # [眼间距比例，眼高比例，眼宽比，眼高比]
        'nose': [nose_height_ratio, nose[1], nose[2]],  # [鼻位置高比例，鼻大小宽比，鼻大小高比]
        'lip': [lip_height_ratio, lip[1], lip[2]]  # [唇位置高比例，唇大小宽比例，唇大小高比]
    }
    return organ_boxes


def read_feature(file_path):
    # step1 Api 获取脸型，五官点阵，是否有眼镜，脸型，性别
    landmark72, angle, gender, glasses, faceshape = get_baseInfo(file_path)

    # step2 数据预处理
    landmark72 = landmark72_trans(landmark72)
    # print(angle)
    if -20 < angle < 20:
        print('-------------Normal--------------')
        pass
    else:
        # angle = angle / 180 * math.pi
        Image.open(file_path).rotate(angle, expand=1).save(file_path)
        landmark72, angle, gender, glasses, faceshape = get_baseInfo(file_path)
        landmark72 = landmark72_trans(landmark72)

    left_eyebrow = point2feature_ebr(landmark72[22:30])
    left_eye = point2feature_eye(landmark72[13:22])
    nose = point2feature_nose(landmark72)
    lip = point2feature_lip(landmark72)
    width = landmark72[12][0] - landmark72[0][0]
    right_eyebrow = point_to_vector(landmark72[39:47])
    right_eye = point_to_vector(landmark72[30:39])
    chin = landmark72[:13] - landmark72[6]

    relative_box = get_relative_position(landmark72)
    # org_struct = org_alignment(organ_struct(landmark72))
    return left_eye, right_eye, left_eyebrow, right_eyebrow, lip, nose, chin, relative_box, width, glasses, faceshape


def compare_feature(org, feature):
    features = scio.loadmat('feature_mat/{}'.format(org))
    target = features['data'][:] - feature
    top_index = np.argmin(np.linalg.norm(target, axis=(1, 2)))
    # print(org, np.linalg.norm(target, axis=(1, 2))[top_index])
    return top_index


def compare_face(faceshape, feature):
    typ = FaceShape[faceshape]
    features = scio.loadmat('feature_mat/face_{}'.format(typ))
    target = features['data'][:] - feature
    top_index = np.argmin(np.linalg.norm(target, axis=(1, 2)))
    # print('11', faceshape, np.linalg.norm(target, axis=(1, 2))[top_index])
    return typ + '-' + str(top_index + 1)


def get_carton_points(feature_index):
    cartoon_points = []
    # 顺序固定
    for org in ['left_eyebrow', 'left_eye', 'nose', 'lip', 'right_eyebrow', 'right_eye']:
        data = CartoonPoint[org][feature_index[org]]
        cartoon_points.append(data)
    return cartoon_points


def merge_all(relative_box, feature_index, face_id):
    cartoon_points = get_carton_points(feature_index)
    image = Image.open(root_dir + '/cartoon/face/{}.png'.format(face_id)).convert('RGBA')

    # organ_boxes = {
    #     'eye_br': [eyebrs_width_ratio, eyebrs_height_ratio, (left_eyebr[1] + right_eyebr[1]) / 2,
    #                (left_eyebr[2] + right_eyebr[2]) / 2],  # [眉间距比例，眉高比例，眉毛宽比，眉毛高比]
    #     'eye': [eyes_width_ratio, eyes_height_ratio, (left_eye[1] + right_eye[1]) / 2,
    #             (left_eye[2] + right_eye[2]) / 2],  # [眼间距比例，眼高比例，眼宽比，眼高比]
    #     'nose': [nose_height_ratio, nose[1], nose[2]],  # [鼻位置高比例，鼻大小宽比，鼻大小高比]
    #     'lip': [lip_height_ratio, lip[1], lip[2]]  # [唇位置高比例，唇大小宽比例，唇大小高比]
    # }

    typ, fid = face_id.split('-')
    face_data = CartoonPoint[typ + '_shape'][int(fid) - 1]

    eyebr_data = relative_box['eye_br']
    eyebr_dis = eyebr_data[0] * 297
    eyebr_y = eyebr_data[1] * 210
    left_eyebr = [face_data[2] - eyebr_dis / 2, face_data[3] - eyebr_y]
    right_eyebr = [face_data[2] + eyebr_dis / 2, face_data[3] - eyebr_y]

    eye_data = relative_box['eye']
    eye_dis = eye_data[0] * 297
    eye_y = eye_data[1] * 210
    left_eye = [face_data[2] - eye_dis / 2, face_data[3] - eye_y]
    right_eye = [face_data[2] + eye_dis / 2, face_data[3] - eye_y]

    nose_data = relative_box['nose']
    nose_y = nose_data[0] * 210
    nose = [face_data[2], face_data[3] - nose_y]

    lip_data = relative_box['lip']
    lip_y = lip_data[0] * 210
    lip = [face_data[2], face_data[3] - lip_y]

    print(face_data)  # [290 265 185 475]
    norm_real_points = np.array([left_eyebr, left_eye, nose, lip, right_eyebr, right_eye])

    boxes = norm_real_points - cartoon_points

    print([int(eye_data[2] * 297), int(eye_data[3] * 210)])
    print([int(eyebr_data[2] * 297), int(eyebr_data[3] * 210)])
    print([int(nose_data[1] * 297), int(nose_data[2] * 210)])
    print([int(lip_data[1] * 297), int(lip_data[2] * 210)])
    # 调整位置和耳朵平齐
    # eye_height = boxes[1][1]
    # cartoon_eye = cartoon_points[1][1] + eye_height
    # print(ear_height - cartoon_eye)
    # boxes = boxes + [0, ear_height - cartoon_eye]
    # print('boxes===', boxes)
    TypOrgans = ['left_eyebrow', 'left_eye', 'nose', 'lip', 'right_eyebrow', 'right_eye']

    norm_real_glasses = (norm_real_points[1] + norm_real_points[-1]) / 2
    # glasses_box = norm_real_glasses + [0, 35] + chin_point - Glasses
    for i, org in enumerate(TypOrgans):
        organ = Image.open(root_dir + "/cartoon/{}/{}.png".format(org, feature_index[org] + 1))
        # if org == 'nose':
        #     print([int(nose_data[1]*297), int(nose_data[2]*210)])
        #     organ = organ.resize([int(nose_data[1]*297), int(nose_data[2]*210)])
        image.paste(organ, list(boxes[i].astype(np.int)), mask=organ)
    # if feature_index['glasses'] == 1:
    #     organ = Image.open(root_dir + "/cartoon/{}/{}_{}.png".format('glasses', 'glasses', 1))
    #     image.paste(organ, list(glasses_box.astype(np.int)), mask=organ)
    return image


def main(file_path, face_id=None, ebr_id=None):
    # step1 获取所有特征数据
    left_eye, _ry, left_eyebrow, _rb, lip, nose, chin, relative_box, width, glasses, faceshape = read_feature(file_path)

    # step2 获取Cartoon匹配序号
    left_eye_id = compare_feature('left_eye', left_eye)
    # right_eye_id = compare_feature('right_eye', right_eye)
    left_eyebrow_id = compare_feature('left_eyebrow', left_eyebrow)
    # right_eyebrow_id = compare_feature('right_eyebrow', right_eyebrow)
    lip_id = compare_feature('lip', lip)
    nose_id = compare_feature('nose', nose)
    face_id = compare_face(faceshape, chin) if not face_id else face_id
    # print(face_id)
    # if ebr_id != None:
    #     left_eyebrow_id = ebr_id

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

    image = merge_all(relative_box, feature_index, face_id)
    # image.save('res' + file_path.replace('jpg', 'png'))
    # image.show()
    return image, feature_index


def final_eye_try():
    face_dir = 'C:/Users/chk01/Desktop/tt'
    # for face in ['A-1', 'A-2', 'A-3', 'A-4', 'B-1', 'B-2', 'B-3', 'B-4', 'C-1', 'C-2', 'C-3', 'C-4', 'E-3']:
    face = 'A-1'
    # 'A-2', 'A-3', 'A-4', 'B-1', 'B-2', 'B-3', 'B-4', 'C-1', 'C-2', 'C-3', 'C-4', 'E-3'
    dir_path = os.listdir(face_dir)
    for file in dir_path:
        if file.endswith('jpg'):
            file_path = face_dir + '/{}'.format(file)
            # if not os.path.exists(file_path.replace('.jpg', '.png')):
            print(file, 'OK')

            image, feature_index = main(file_path, face)
            image.save(file_path.replace('.jpg', '.png'))
            with open(file_path.replace('jpg', 'txt'), 'a') as text_file:
                text_file.writelines('---------------------------------\n')
                for org, item in feature_index.items():
                    if org != 'chin':
                        text_file.writelines(org + ':' + str(int(item) + 1) + '\n')


if __name__ == "__main__":
    final_eye_try()
    # face_dir = 'C:/Users/chk01/Desktop/eye_test/2.jpg'
    # for i in range(25):
    #     image, feature_index = main(face_dir, face_id="A-1", ebr_id=i)
    #     image.save(face_dir.replace('2.jpg', str(feature_index['left_eyebrow'] + 1) + '.png'))
