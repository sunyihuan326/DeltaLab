# coding:utf-8
'''
Created on 2017/12/20.

@author: chk01
'''
import os
import scipy.io as scio

from practice_one.Company.match_check.utils import *

org = 'lip'

root_dir = 'C:/Users/chk01/Desktop/Delta/image'
save_dir = '../load_material/feature_mat/' + org


def p2f(landmarks):
    point1 = [landmarks[58], landmarks[59], landmarks[60], landmarks[61],
              landmarks[68], landmarks[67], landmarks[66]]
    x1 = [p[0] for p in point1]
    y1 = [p[1] for p in point1]
    wid1 = max(x1) - min(x1)
    hei1 = max(y1) - min(y1)

    center1 = landmarks[67]
    # feature1 = (point1 - center1) / np.array([wid1, hei1])
    feature1 = (point1 - center1)

    point2 = [landmarks[58], landmarks[65], landmarks[64], landmarks[63],
              landmarks[71], landmarks[70], landmarks[69]]
    x2 = [p[0] for p in point2]
    y2 = [p[1] for p in point2]
    wid2 = max(x2) - min(x2)
    hei2 = max(y2) - min(y2)

    center2 = landmarks[70]
    # feature2 = (point2 - center2) / np.array([wid2, hei2])
    feature2 = (point2 - center2)
    feature = np.zeros([14, 2])
    feature[:7, :] = feature1
    feature[7:, :] = feature2

    return feature


def compare_feature(feature):
    features = scio.loadmat(save_dir)
    target = features['data'][:] - feature
    top_index = np.argmin(np.linalg.norm(target, axis=(1, 2)))
    score = round(np.linalg.norm(target, axis=(1, 2))[top_index], 2)
    return top_index, score


def get_point_feature():
    print('开始{}导入'.format(org))
    dir_path = os.listdir(root_dir + '/src/' + org)
    m = len(dir_path)
    n = 14
    data = np.zeros([m, n, 2])
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0]) - 1
        full_path = root_dir + '/src/' + org + '/' + sourceDir
        landmark72 = get_baseInfo(full_path)
        landmark72 = landmark72_trans(landmark72)
        feature = p2f(landmark72)
        data[_id] = feature
        print('load--->{}---图{}'.format(org, _id))
    scio.savemat(save_dir, {"data": data})
    print('完成{}导入'.format(org))


def main(file):
    landmark72 = get_baseInfo(file)

    landmark72 = landmark72_trans(landmark72)

    feature = p2f(landmark72)
    cid, score = compare_feature(feature)
    print('output::', cid + 1, 'diff::', score)
    print('----------------------')


def check_load_correct():
    for i in range(20):
        print('input::', i + 1)
        file = root_dir + '/src/lip/{}.jpg'.format(i + 1)
        main(file)


if __name__ == '__main__':
    # get_point_feature()
    check_load_correct()

    # file = '3.jpg'
    # main(file)
