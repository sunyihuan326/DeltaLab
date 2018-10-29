# coding:utf-8
'''
Created on 2017/12/20.

@author: chk01
'''
from practice_one.Company.load_material.utils import *


def get_point_feature():
    print('开始{}导入'.format('eye'))
    dir_path = os.listdir('material/cartoon/left_eye/model')
    m = 35
    n = 7
    X = np.zeros([m, n, 2]) + 999
    Y = np.zeros([m, 1]) + 999
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0])
        full_path = 'material/cartoon/left_eye/model/' + sourceDir
        landmark72, _, _, _, _ = get_baseInfo(full_path)
        landmark72 = landmark72_trans(landmark72)
        feature = point2feature_eye(landmark72)
        X[_id - 1] = feature
        Y[_id - 1] = _id
        print('load--->{}---图{}'.format('eye', _id))
    scio.savemat('material/feature_matrix/eye', {"X": X, "Y": Y})
    print('完成{}导入'.format('eye'))


if __name__ == '__main__':
    get_point_feature()
