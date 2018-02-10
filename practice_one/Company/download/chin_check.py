# coding:utf-8
'''
Created on 2017/12/27.

@author: chk01
'''
from practice_one.Company.load_material.utils import *


def get_face_feature():
    for typ in ['A','B']:
        print('开始{}型导入'.format(typ))
        dir_path = os.listdir('material/cartoon/face/{}/model'.format(typ))
        m = 25
        n = 13
        X = np.zeros([m, n, 2]) + 999
        Y = np.zeros([m, 1]) + 999
        for i, sourceDir in enumerate(dir_path):
            _id = int(sourceDir.split('.')[0].split('-')[1])
            full_path = 'material/cartoon/face/{}/model/'.format(typ) + sourceDir
            landmark72, _, _, _, _ = get_baseInfo(full_path)
            landmark72 = landmark72_trans(landmark72)
            _data = point2feature_chin(landmark72)
            X[_id - 1] = _data
            Y[_id - 1] = _id
            print('load--->{}---图{}'.format(typ, _id))
        scio.savemat('material/feature_matrix/chin_' + typ, {"X": X, "Y": Y})
        print('完成{}导入'.format(typ))


if __name__ == '__main__':
    get_face_feature()
