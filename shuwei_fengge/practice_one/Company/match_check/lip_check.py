# coding:utf-8
'''
Created on 2017/12/20.

@author: chk01
'''

from practice_one.Company.load_material.utils import *

org = 'lip'

root_dir = 'C:/Users/chk01/Desktop/Delta/image'
save_dir = '../load_material/feature_matrix/' + org

org_data = scio.loadmat('../load_material/feature_matrix/{}'.format(org))
org_ob = NearestNeighbor()
org_ob.train(X=org_data['X'], Y=org_data['Y'])


def compare_feature(feature):
    org_id = org_ob.predict(feature)
    return org_id


def get_point_feature():
    print('开始{}导入'.format(org))
    dir_path = os.listdir(root_dir + '/src/' + org)
    m = 20
    n = 14
    X = np.zeros([m, n, 2]) + 999
    Y = np.zeros([m, 1]) + 999
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0]) - 1
        full_path = root_dir + '/src/' + org + '/' + sourceDir
        landmark72, _, _, _, _ = get_baseInfo(full_path)
        landmark72 = landmark72_trans(landmark72)
        feature = point2feature_lip(landmark72)
        X[_id] = feature
        Y[_id] = _id + 1
        # p2f(landmark72[39:47])
        print('load--->{}---图{}'.format(org, _id + 1))
    scio.savemat(save_dir, {"X": X, "Y": Y})
    print('完成{}导入'.format(org))


def main(file):
    landmark72, _, _, _, _ = get_baseInfo(file)

    landmark72 = landmark72_trans(landmark72)

    feature = point2feature_lip(landmark72)
    cid = compare_feature(feature)
    print('output::', cid)
    print('----------------------')


def check_load_correct():
    for i in range(20):
        print('input::', i + 1)
        file = root_dir + '/src/lip/{}.jpg'.format(i + 1)
        main(file)


if __name__ == '__main__':
    get_point_feature()
    # check_load_correct()

    # file = '1003.jpg'
    # main(file)
