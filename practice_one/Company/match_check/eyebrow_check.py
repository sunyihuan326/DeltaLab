# coding:utf-8
'''
Created on 2017/12/20.

@author: chk01
'''
import os
import scipy.io as scio

# from practice_one.Company.match_check.utils import *
from practice_one.Company.load_material.urils import *

org = 'left_eyebrow'

root_dir = 'C:/Users/chk01/Desktop/Delta/image'
save_dir = '../load_material/feature_matrix/' + org


def p2f(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    center = (points[0] + points[4]) / 2
    feature = (points - center) / np.array([wid, hei])
    # feature = points - center
    return feature


def compare_feature(feature):
    eyebr_data = scio.loadmat('../load_material/feature_matrix/eyebr')
    eyebr = NearestNeighbor()
    eyebr.train(X=eyebr_data['X'], Y=eyebr_data['Y'])
    sample_num = eyebr.predict(feature)
    return sample_num


def get_point_feature():
    print('开始{}导入'.format(org))
    dir_path = os.listdir(root_dir + '/src/' + org)
    m = 25
    n = 8
    X = np.zeros([m, n, 2])
    Y = np.zeros([m, ])
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0]) - 1
        full_path = root_dir + '/src/' + org + '/' + sourceDir
        landmark72 = get_baseInfo(full_path)
        landmark72 = landmark72_trans(landmark72)
        feature = p2f(landmark72[22:30])
        X[_id] = feature
        Y[_id] = _id + 1
        # p2f(landmark72[39:47])
        print('load--->{}---图{}'.format(org, _id))
    scio.savemat(save_dir, {"X": X, "Y": Y})
    print('完成{}导入'.format(org))


def main(file):
    landmarks, _, _, _, _ = get_baseInfo(file)
    landmark72 = landmark72_trans(landmarks)

    left_eyebrow = p2f(landmark72[22:30])
    cid = compare_feature(left_eyebrow)
    print('output::', cid)
    print('----------------------')


def check_load_correct():
    for i in range(25):
        print('input::', i + 1)
        file = root_dir + '/src/left_eyebrow/{}.jpg'.format(i + 1)
        try:
            main(file)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    # get_point_feature()
    check_load_correct()

    # file = '1003.jpg'
    # main(file)
