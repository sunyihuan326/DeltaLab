# coding:utf-8
'''
Created on 2017/12/20.

@author: chk01
'''
import os
from practice_one.Company.match_check.utils import *
import matplotlib.pyplot as plt
check_dir = 'C:/Users/chk01/Desktop/Delta/image/check/src'


def main():
    org = 'left_eye'
    dir_path = os.listdir(check_dir + '/' + org)
    for sourceDir in dir_path:
        full_path = check_dir + '/' + org + '/' + sourceDir
        landmark72 = get_baseInfo(full_path)
        landmark72 = landmark72_trans(landmark72)
        points = landmark72[13:21]
        plt.plot(points[:, 0], points[:, 1], 'o-')
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
        assert 1==0
        # for org in ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow', 'lip', 'nose']:
        #     oldname = "../load_material/cartoon/{}/{}.png".format(org, feature_index[org] + 1)
        #     shutil.copyfile(oldname, face_dir + '/{}_{}.png'.format(org, feature_index[org] + 1))


if __name__ == '__main__':
    main()
