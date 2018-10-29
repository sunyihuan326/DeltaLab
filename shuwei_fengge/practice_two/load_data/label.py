# coding:utf-8
'''
Created on 2017/12/5.

@author: chk01
'''
from shuwei_fengge.practice_two.load_data.utils import *
import operator


def get_face_box(points):
    X = points[:, 0]
    Y = points[:, 1]
    min_x = min(X)
    max_x = max(X)
    min_y = min(Y)
    max_y = max(Y)
    wid = max(max_y - min_y, max_x - min_x)

    new_x = min_x - (wid - (max_x - min_x)) // 2
    new_y = min_y - (wid - (max_y - min_y)) // 2
    region = [new_x, new_y, new_x + wid, new_y + wid]
    return region


def main():
    resList = fetch_source_data()
    num = len(resList)
    print('total num ------>', num)

    for i, res in enumerate(resList):
        point_dir = '../data/label/Image-Label-{}.mat'.format(res['_id'])
        sorted_x = sorted(res['t_style_text'].items(), key=operator.itemgetter(1))
        label = convert_to_one_hot(int(sorted_x[-1][0].replace("t_style_text_", "")), 9)
        scio.savemat(point_dir, {'Label': label})


if __name__ == '__main__':
    main()
