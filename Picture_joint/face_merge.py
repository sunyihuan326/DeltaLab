# coding:utf-8
'''
Created on 2017/11/24

@author: sunyihuan
'''
from PIL import Image
import numpy as np
import pandas as pd

data = pd.read_csv("FaceData.csv")
face1 = eval(data["FaceData"][0])
print(face1)
# image_val[min_y:min_y + hei, min_x:min_x + wid, :] = organ_val
# image.show()
# image.save('2.png')


#
# face_norm = {
#     'face': {'center': [1, 2], 'size': [600, 919]},
#     'lip': {'center': [1, 2], 'size': [30, 30]},
#     'nose': {'center': [1, 2], 'size': [30, 30]},
#     'eyebrow': {'center': [1, 2], 'size': [600, 919]},
# }
per = {'left_eye': {'size': [45 / 175, 77 / 233], 'center': [45 / 175, 98 / 233]},
       'left_eyebrow': {'size': [209 / 175, 57 / 233], 'center': [39 / 175, 77 / 233]},
       'right_eye': {'size': [167 / 175, 77 / 233], 'center': [451 / 175, 358 / 233]},
       'nose': {'size': [172 / 175, 103 / 233], 'center': [87 / 175, 159 / 233]},
       'right_eyebrow': {'size': [211 / 175, 57 / 233], 'center': [462 / 175, 284 / 233]},
       'lip': {'size': [200 / 175, 66 / 233], 'center': [87 / 175, 183 / 233]},
       'center': [88 / 175, 117 / 233]}

# 环环数据
# left_eye = {'type': 'right_eye', 'center': [141, 436], 'size': [167, 77]}
# left_eyebrow = {'type': 'left_eyebrow', 'center': [141, 328], 'size': [209, 57]}
# lip = {'type': 'lip', 'center': [299, 750], 'size': [200, 66]}
# nose = {'type': 'nose', 'center': [299, 658], 'size': [172, 103]}
# glass = {'type': 'nose', 'center': [299, 469], 'size': [172, 103]}


# 燕子数据：
left_eye = {'type': 'right_eye', 'center': [451, 358], 'size': [167, 77]}
left_eyebrow = {'type': 'left_eyebrow', 'center': [140, 284], 'size': [209, 57]}
lip = {'type': 'lip', 'center': [297, 672], 'size': [200, 66]}
nose = {'type': 'nose', 'center': [297, 522], 'size': [172, 103]}

all_organ = {
    'eye': left_eye,
    'eyebrow': left_eyebrow,
    'lip': lip,
    'nose': nose
    # 'glass': glass
}
# 标准脸宽高
face_size = [175, 233]


# nose = {'type': 'nose', 'size': [172, 103], 'center': [297, 522]}


def get_box(face1, org):
    '''
    :param face1:face ID
    :param organ:五官中心点数据
    :return: box位置
    '''
    organ = all_organ[org]
    # 偏移
    res = np.array(face1[organ['type']]['center']) - np.array(
        np.array(per[organ['type']]['center']) * np.array(face_size))
    # print(np.array(per[organ['type']]['center']) * np.array(face_size))
    return list(res.astype(np.int32))


def merge_all(feature_index, face=1):
    '''
    :param feature_index: dict={'lip':1,'nose':2,'left_eye':2,'right_eye':3,'left_eyebrow':4,'right_eyebrow':4}
    :param glasses:int 0:无,1:有
    :param face:int 1
    :return:
    '''

    # 底图
    target = Image.new('RGBA', (400, 500), (0, 0, 0, 0))

    image = Image.open('./cartoon/face/face_{}.png'.format(face))
    image.thumbnail(face_size)
    main_bias = np.array([200, 250]) - np.array(face1['center'])
    target.paste(image, list(main_bias), mask=image)
    # target.show()

    TypOrgans = ['lip', 'nose', 'eye', 'eyebrow']
    if feature_index['glasses'] == 1:
        TypOrgans.append('glasses')

    for org in TypOrgans:
        organ = Image.open("./cartoon/{}/{}_{}.png".format(org, org, feature_index[org]))
        # p = np.min(np.array(face_size) / np.array(organ.size))
        organ.thumbnail(face_size)
        org_box = get_box(face1, org)
        print(org_box)
        target.paste(organ, list(np.array(org_box) + main_bias), mask=organ)
    return target




if __name__ == '__main__':
    feature_index = {
        'eye': 1,
        'eyebrow': 1,
        'lip': 3,
        'nose': 3,
        'glasses': 0,
    }
    image = merge_all(feature_index, 3)
    image.show()
