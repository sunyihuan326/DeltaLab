# coding:utf-8
'''
Created on 2017/12/20.

@author: chk01
'''
from practice_one.Company.load_material.utils import *

org = 'left_eye'
root_dir = 'C:/Users/chk01/Desktop/Delta/image'
save_dir = '../load_material/feature_matrix/' + org

matrix = 'eye'

org_data = scio.loadmat('../load_material/feature_matrix/{}'.format(matrix))
org_ob = NearestNeighbor()
org_ob.train(X=org_data['X'], Y=org_data['Y'])


def compare_feature(feature):
    org_id = org_ob.predict(feature)
    return org_id


def get_point_feature():
    print('开始{}导入'.format(org))
    dir_path = os.listdir(root_dir + '/src/' + org)
    m = 32
    n = 7
    X = np.zeros([m, n, 2]) + 999
    Y = np.zeros([m, 1]) + 999
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0]) - 1
        full_path = root_dir + '/src/' + org + '/' + sourceDir
        landmark72, _, _, _, _ = get_baseInfo(full_path)
        landmark72 = landmark72_trans(landmark72)
        feature = point2feature_eye(landmark72)
        X[_id] = feature
        Y[_id] = _id + 1
        # p2f(landmark72[39:47])
        print('load--->{}---图{}'.format(org, _id + 1))
    scio.savemat(save_dir.replace(org, matrix), {"X": X, "Y": Y})
    print('完成{}导入'.format(org))


def main(file):
    landmark72, _, _, _, _ = get_baseInfo(file)

    landmark72 = landmark72_trans(landmark72)

    feature = point2feature_eye(landmark72)
    cid = compare_feature(feature)
    print('output::', cid)
    print('----------------------')


def check_load_correct():
    for i in range(32):
        print('input::', i + 1)
        try:
            file = root_dir + '/src/left_eye/{}.jpg'.format(i + 1)
            main(file)
        except:
            print('Error', file)


if __name__ == '__main__':
    get_point_feature()
    # check_load_correct()

    # file = '1001.jpg'
    # main(file)
