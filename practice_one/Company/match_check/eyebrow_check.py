# coding:utf-8
'''
Created on 2017/12/20.

@author: chk01
'''
from practice_one.Company.load_material.utils import *

org = 'left_eyebrow'

root_dir = 'C:/Users/chk01/Desktop/Delta/image'
save_dir = '../load_material/feature_matrix/' + org


matrix = 'eyebr'
org_data = scio.loadmat('../load_material/feature_matrix/{}'.format(matrix))
org_ob = NearestNeighbor()
org_ob.train(X=org_data['X'], Y=org_data['Y'])


def compare_feature(feature):
    org_id = org_ob.predict(feature)
    return org_id


def get_point_feature():
    print('开始{}导入'.format(org))
    dir_path = os.listdir(root_dir + '/src/' + org)
    m = 25
    n = 8
    X = np.zeros([m, n, 2])
    Y = np.zeros([m, 1])
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0]) - 1
        full_path = root_dir + '/src/' + org + '/' + sourceDir
        landmark72, _, _, _, _ = get_baseInfo(full_path)
        landmark72 = landmark72_trans(landmark72)
        feature = point2feature_ebr(landmark72)
        X[_id] = feature
        Y[_id] = _id + 1
        # p2f(landmark72[39:47])
        print('load--->{}---图{}'.format(org, _id))
    scio.savemat(save_dir, {"X": X, "Y": Y})
    print('完成{}导入'.format(org))


def main(file):
    landmarks, _, _, _, _ = get_baseInfo(file)
    landmark72 = landmark72_trans(landmarks)

    left_eyebrow = point2feature_ebr(landmark72)
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
    # check_load_correct()

    file = '1003.jpg'
    main(file)
