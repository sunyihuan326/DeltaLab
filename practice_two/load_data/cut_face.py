# coding:utf-8
'''
Created on 2017/12/2.

@author: chk01
'''
from practice_two.load_data.utils import *
from PIL import Image
import os


def get_face_box(points):
    X = points[:, 0]
    Y = points[:, 1]
    min_x = min(X)
    max_x = max(X)
    min_y = min(Y)
    max_y = max(Y)
    wid = max(max_y - min_y, max_x - min_x)
    wid = 1.8 * wid
    new_x = min_x - (wid - (max_x - min_x)) // 2
    new_y = min_y - (wid - (max_y - min_y)) // 2
    p = 0.2
    region = [new_x, new_y - p * wid, new_x + wid, new_y + (1 - p) * wid]
    return region


def main():
    dir_path = os.listdir('C:/Users/chk01/Desktop/top_head')
    for file in dir_path:
        print(file)
        if not os.path.exists('C:/Users/chk01/Desktop/top_head/cut/' + file):
            img_path = 'C:/Users/chk01/Desktop/top_head/' + file
            image = Image.open(img_path).convert("RGB")
            points = get_landmark72(img_path)
            region = get_face_box(points)
            res = image.crop(region).resize([64, 64])
            res.save('C:/Users/chk01/Desktop/top_head/cut/' + file)


if __name__ == '__main__':
    main()
    # pass
