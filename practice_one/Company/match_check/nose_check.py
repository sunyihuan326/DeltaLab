# coding:utf-8
'''
Created on 2017/12/20.

@author: chk01
'''
import os
import scipy.io as scio

from practice_one.Company.match_check.utils import *

org = 'nose'

root_dir = 'C:/Users/chk01/Desktop/Delta/image'
save_dir = '../load_material/feature_matrix/' + org


def p2f(landmarks):
    points = landmarks[49:55]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    center = landmarks[57]
    feature = (points - center) / np.array([wid, hei])
    # feature = points - center
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
    n = 6
    X = np.zeros([m, n, 2])
    Y = np.zeros([m, ])
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0]) - 1
        full_path = root_dir + '/src/' + org + '/' + sourceDir
        landmark72 = get_baseInfo(full_path)
        landmark72 = landmark72_trans(landmark72)
        feature = p2f(landmark72)
        X[_id] = feature
        Y[_id] = _id + 1
        # p2f(landmark72[39:47])
        print('load--->{}---图{}'.format(org, _id))
    scio.savemat(save_dir, {"X": X, "Y": Y})
    print('完成{}导入'.format(org))


def main(file):
    landmark72 = get_baseInfo(file)

    landmark72 = landmark72_trans(landmark72)

    feature = p2f(landmark72)
    cid, score = compare_feature(feature)
    print('output::', cid + 1, 'diff::', score)
    print('----------------------')

    # right_eyebrow = p2f(landmark72[39:47])


def check_load_correct():
    for i in range(12):
        print('input::', i + 1)
        file = root_dir + '/src/nose/{}.jpg'.format(i + 1)
        main(file)


if __name__ == '__main__':
    get_point_feature()
    # check_load_correct()

    # file = '3.jpg'
    # main(file)
