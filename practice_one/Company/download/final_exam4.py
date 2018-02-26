# coding:utf-8
'''
Created on 2017/12/27.

@author: chk01
'''
from practice_one.Company.load_material.utils import *

eyebr, eye, nose, lip, chinA, chinB, chinC, chinD, chinE = load_feature_matrix()

ChinData = {"A": chinA, "B": chinB, "C": chinC, "D": chinD, "E": chinE}
CartoonPoint = load_cartoon_center()
Glasses = [164, 75]


def organ_struct(landmark72):
    # reb,reye,nose,lip,chin,leb,leye
    return [(landmark72[24] + landmark72[28]) / 2, landmark72[21],
            (landmark72[51] + landmark72[52]) / 2, landmark72[60],
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
    landmark72, angle, gender, glasses, faceshape = get_baseInfo(file_path)
    # # 图片矫正待优化
    if -1 < angle < 1:
        pass
    else:
        # import math
        print('angle', angle)
        # angle = -angle / 180 * math.pi
        Image.open(file_path).rotate(angle, expand=1).save(file_path)
        landmark72, angle, gender, glasses, faceshape = get_baseInfo(file_path)

    # step2 数据预处理
    landmark72 = landmark72_trans(landmark72)
    faceshape = get_real_faceshape(faceshape)

    # tran_matrix = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
    # landmark72 = np.matmul(landmark72-landmark72[6], tran_matrix)+landmark72[6]

    eyebr = point2feature_ebr(landmark72)
    eye = point2feature_eye(landmark72)
    nose = point2feature_nose(landmark72)
    lip = point2feature_lip(landmark72)
    chin = point2feature_chin(landmark72)

    width = landmark72[12][0] - landmark72[0][0]
    height = np.abs((landmark72[12][1] + landmark72[0][1]) / 2 - landmark72[6][1])

    org_struct = org_alignment(organ_struct(landmark72))
    return eyebr, eye, nose, lip, chin, org_struct, width, height, glasses, faceshape


def get_carton_points(feature_index):
    cartoon_points = []
    # 顺序固定
    for org in ['left_eyebrow', 'left_eye', 'nose', 'lip', 'chin', 'right_eyebrow', 'right_eye']:
        if org == 'chin':
            typ, _id = feature_index[org].split('-')
            data = CartoonPoint[typ + '_shape'][int(_id) - 1][2:4]
        else:
            data = CartoonPoint[org][feature_index[org] - 1][:2]
        cartoon_points.append(data)
    return cartoon_points


def eye_dis_check(position, fw):
    out_position = position
    eye_dis_in = position[-1][0] - position[1][0]
    eye_ratio_in = eye_dis_in / fw
    eye_ratio_min = 0.43
    # eye_ratio_max = 0.485
    eye_ratio_max = 0.45
    eye_ratio = min(max(eye_ratio_in, eye_ratio_min), eye_ratio_max)

    eye_dis_min = 127
    eye_dis_max = 130
    eye_dis_out_1 = eye_ratio * fw
    eye_dis_out = min(max(eye_dis_out_1, eye_dis_min), eye_dis_max)
    # 126.42
    print('eye_dis_out', eye_dis_out)

    eye_diff = eye_dis_out - eye_dis_in
    out_position[-1][0] += eye_diff / 2
    out_position[1][0] -= eye_diff / 2

    out_position[-2][0] += eye_diff / 2
    out_position[0][0] -= eye_diff / 2

    return out_position


def nose_dis_check(position):
    out_position = position
    brow_hei = out_position[1][1] - out_position[4][1]
    nose_hei = out_position[2][1] - out_position[4][1]
    nose_ratio_in = nose_hei / brow_hei
    nose_ratio_min = 0.47
    nose_ratio_max = 0.615
    print('nose_ratio_in', nose_ratio_in)
    if nose_ratio_in < nose_ratio_min:
        print('nose_check_min')
        nose_ratio = nose_ratio_min
    elif nose_ratio_in > nose_ratio_max:
        print('nose_check_max')
        nose_ratio = nose_ratio_max
    else:
        nose_ratio = nose_ratio_in

    nose_ratio = 0.565
    nose_hei_out = nose_ratio * brow_hei
    out_position[2][1] = nose_hei_out + out_position[4][1]
    return out_position


def lip_dis_check(position, hou):
    # print(hou)
    # leb,leye,nose,lip,chin,reb,reye
    out_position = position
    nose2lip = out_position[3][1] - out_position[2][1]
    # lip2chin = out_position[4][1] - hou - out_position[3][1]
    nose_hei = out_position[4][1] - out_position[2][1]
    lip_ratio_in = nose2lip / nose_hei
    lip_ratio_min = 0.26
    lip_ratio_max = 0.27
    print('lip_ratio_in', lip_ratio_in)
    if lip_ratio_in < lip_ratio_min:
        print('lip_check_min')
        lip_ratio = lip_ratio_min
    elif lip_ratio_in > lip_ratio_max:
        print('lip_check_max')
        lip_ratio = lip_ratio_max
    else:
        lip_ratio = lip_ratio_in

    out_position[3][1] = out_position[2][1] + lip_ratio * nose_hei

    return out_position


def eyebrow_hei_check(position, hei):
    out_position = position
    ratio_in = (out_position[1][1] - out_position[0][1]) / hei
    ratio_max = 0.20
    ratio_min = 0.17
    ratio_out = min(max(ratio_in, ratio_min), ratio_max)

    dis_in = ratio_out * hei
    dis_min = 45
    dis_max = 48
    dis_out = min(max(dis_in, dis_min), dis_max)
    print(dis_out)
    print(ratio_out)
    out_position[0][1] = out_position[1][1] - dis_out
    out_position[-2][1] = out_position[1][1] - dis_out
    # 0.2286 ok
    # 0.14 close
    # 0.25 far
    # 0.239 far
    print('----222----', (out_position[1] - out_position[0])[1])
    return out_position


def merge_all(real_width, real_height, real_points, feature_index):
    skin_color = 1
    face_id = feature_index['chin']
    typ, fid = face_id.split('-')

    cartoon_points = get_carton_points(feature_index)
    image = Image.open('material/cartoon/face/{}/{}/{}.png'.format(typ, skin_color, face_id)).convert('RGBA')

    face_data = CartoonPoint[typ + '_shape'][int(fid) - 1]
    ratio_x = face_data[0] / real_width
    # print(ratio_x)
    ear_height = face_data[1]
    chin_point = cartoon_points[4]
    real_chin_point = real_points[4]

    ratio_y = (ear_height - chin_point[1]) / (np.array(real_points) - real_chin_point)[1][1]
    # ratio_y = 250/real_height
    norm_real_points = (np.array(real_points) - real_chin_point) * [ratio_x, ratio_y]
    # leb,leye,nose,lip,chin,reb,reye
    last_position = norm_real_points + chin_point
    last_position = eye_dis_check(last_position, face_data[0])

    last_position = eyebrow_hei_check(last_position, ear_height)

    last_position = nose_dis_check(last_position)
    hou = CartoonPoint['lip'][feature_index['lip'] - 1][2]
    last_position = lip_dis_check(last_position, hou)
    # boxes = norm_real_points - cartoon_points + chin_point
    boxes = last_position - cartoon_points

    # leb,leye,nose,lip,chin,reb,reye
    # boxes[0] += [0, 6]
    # boxes[1] += [-3, 0]
    # boxes[-1] += [3, 0]
    # boxes[0] += [10, 0]
    # boxes[-2] += [0, 6]
    # boxes[3] += [0, 6]
    # boxes[-1] += [10, 35]
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
            file_path = "material/cartoon/{}/{}.png".format(org, feature_index[org])
            file_path2 = "material/cartoon/{}/{}/{}.png".format(org, skin_color, feature_index[org])
            if org in ['left_eye', 'right_eye', 'nose']:
                file_path = file_path2
            organ = Image.open(file_path)
            image.paste(organ, list(boxes[i].astype(np.int)), mask=organ)

    if feature_index['glasses'] == 1:
        organ = Image.open("material/cartoon/glasses/glasses_1.png")
        image.paste(organ, list(glasses_box.astype(np.int)), mask=organ)
        # image.alpha_composite(organ,tuple(glasses_box.astype(np.int)))

    back = Image.open('material/cartoon/face/{}/{}/{}.png'.format(typ, skin_color, face_id)).convert('RGBA')
    back.alpha_composite(image)
    return back, [face_data[2:4], face_data[4:]]


def main(file_path):
    # step1 获取所有特征数据
    _eyebr, _eye, _nose, _lip, _chin, org_struct, width, height, glasses, faceshape = read_feature(file_path)

    # step2 获取Cartoon匹配序号
    eyebr_id = eyebr.predict(_eyebr)
    eye_id = eye.predict(_eye)
    lip_id = lip.predict(_lip)
    nose_id = nose.predict(_nose)
    # eye_id = 5
    # nose_id = 11
    # lip_id = 11
    # eyebr_id=1

    chin_id = faceshape + '-' + str(ChinData[faceshape].predict(_chin))
    # chin_id = 'B-8'
    # chin_id = 'A-9'
    feature_index = {
        'left_eye': eye_id,
        'right_eye': eye_id,
        'left_eyebrow': eyebr_id,
        'right_eyebrow': eyebr_id,
        'lip': lip_id,
        'nose': nose_id,
        'glasses': glasses,
        'chin': chin_id
    }
    image, pp = merge_all(width, height, org_struct, feature_index)

    # step4 去污（可优化）
    # back = Image.open('A-77.png').convert('RGBA')
    # back = Image.open(root_dir + '/cartoon/face/{}.png'.format(chin_id)).convert('RGBA')
    # back.alpha_composite(image)

    return image, feature_index


def one_file(file):
    image, feature_index = main(file)
    image.save('res.png')
    # image.show()
    print('---------------------------')
    print('-----eyebrow-----:', feature_index['left_eyebrow'], '号素材')
    print('-------eye-------:', feature_index['left_eye'], '号素材')
    print('------nose-------:', feature_index['nose'], '号素材')
    print('-------lip-------:', feature_index['lip'], '号素材')
    print('------chin-------:', feature_index['chin'], '号素材')
    print('-----glasses-----:', feature_index['glasses'])
    print('---------------------------')


def one_dir(dir):
    face_dir = dir
    dir_path = os.listdir(face_dir)
    for file in dir_path:
        file_path = face_dir + '/{}'.format(file)
        if file.endswith('png') and not file.endswith('-res.png'):
            if not os.path.exists(file_path.replace('.png', '-res.png')):
                # print(file, 'OK')
                image, feature_index = main(file_path)
                print('nose', feature_index['nose'])
                print('lip', feature_index['lip'])
                image.save(file_path.replace('.png', '-res.png'))
                # with open(file_path.replace('png', 'txt'), 'a') as text_file:
                #     text_file.writelines('---------------------------------\n')
                #     for org, item in feature_index.items():
                #         text_file.writelines(org + ':' + str(item) + '\n')


if __name__ == "__main__":
    i = 13
    file = 'check/{}.jpg'.format(i)
    one_file(file)
