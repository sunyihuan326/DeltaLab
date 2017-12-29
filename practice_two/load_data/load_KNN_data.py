# coding:utf-8
'''
Created on 2017/12/29.

@author: chk01
'''
from practice_one.Company.load_material.utils import *

LabelToSense = [0, 1, 2, 0, 1, 2, 0, 1, 2]
LabelToOutline = [0, 0, 0, 1, 1, 1, 2, 2, 2]


def get_outline_feature():
    logdir = '../data/image3channel'
    label_dir = '../data/label'
    dir_path = os.listdir(logdir)
    num = len(dir_path)
    print('total num ------>', num)
    n = 13
    X = np.zeros([num, n, 2])
    Y = np.zeros([num, 1])

    for i, file in enumerate(dir_path):
        print('read_{}_data------->loading----->start'.format(file))
        points = scio.loadmat(logdir + '/' + file)['Points']
        label = np.argmax(scio.loadmat(label_dir + '/' + file.replace('Point', 'Label'))['Label'])

        _data = point2feature_chin(points)
        X[i] = _data
        Y[i] = LabelToOutline[label]
        print('load--->{}---图{}'.format('outline', i))
    scio.savemat('knn-outline', {"X": X, "Y": Y})
    print('完成导入')


if __name__ == '__main__':
    get_outline_feature()
