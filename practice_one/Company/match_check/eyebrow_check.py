# coding:utf-8
'''
Created on 2017/12/20.

@author: chk01
'''
import os
import scipy.io as scio

from practice_one.Company.match_check.utils import *

org = 'left_eyebrow'

root_dir = 'C:/Users/chk01/Desktop/Delta/image'
save_dir = '../load_material/feature_mat/' + org


def p2f(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    center = (points[0] + points[4]) / 2
    # feature = (points - center) / np.array([wid, hei])
    feature = points - center
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
    n = 8
    data = np.zeros([m, n, 2])
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0]) - 1
        full_path = root_dir + '/src/' + org + '/' + sourceDir
        landmark72 = get_baseInfo(full_path)
        landmark72 = landmark72_trans(landmark72)
        feature = p2f(landmark72[22:30])
        data[_id] = feature
        print('load--->{}---图{}'.format(org, _id))
    scio.savemat(save_dir, {"data": data})
    print('完成{}导入'.format(org))


def main(file):
    landmark72 = get_baseInfo(file)

    landmark72 = landmark72_trans(landmark72)

    left_eyebrow = p2f(landmark72[22:30])
    cid, score = compare_feature(left_eyebrow)
    print('output::', cid + 1, 'diff::', score)
    print('----------------------')


def check_load_correct():
    for i in range(25):
        print('input::', i + 1)
        file = root_dir + '/src/left_eyebrow/{}.jpg'.format(i + 1)
        main(file)


if __name__ == '__main__':
    # get_point_feature()
    check_load_correct()

    # file = '3.jpg'
    # main(file)
