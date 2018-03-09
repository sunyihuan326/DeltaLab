# coding:utf-8 
'''
created on 2018/2/28

@author:Dxq
'''
import os
from PIL import Image, ImageEnhance
from eyelid_model.baidu import get_baseInfo
from tqdm import tqdm

# from skimage import feature

TYPE = 'double'
FILE_DIR = 'C:/Users/chk01/Desktop/eyelid/test/' + TYPE
SAVE_DIR = 'C:/Users/chk01/Desktop/eyelid/test/new/' + TYPE
SAMPLE_SIZE = 64


def main():
    pass


def baidu_service(file_path):
    _, landmark72, angle = get_baseInfo(file_path)
    if -2 < angle < 2:
        pass
    else:
        print('angle', angle)
        Image.open(file_path).rotate(angle, expand=1).save(file_path)
        _, landmark72, angle = get_baseInfo(file_path)
    return landmark72


def eye_region(eye_points):
    x = eye_points[:, 0]
    y = eye_points[:, 1]
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    wid = max((max_y - min_y), (max_x - min_x)) + 50
    region = ((max_x + min_x - wid) // 2, (max_y + min_y - wid) // 2, (max_x + min_x + wid) // 2,
              (max_y + min_y + wid) // 2)
    return region


def pre_do():
    # for i in range(1, 5):
    #     enhancer = ImageEnhance.Contrast(left_eye)
    #     left_eye = enhancer.enhance(1 + i * 0.2)
    #     for j in range(1, 5):
    #         brighter = ImageEnhance.Brightness(left_eye)
    #         left_eye = brighter.enhance(1 + j * 0.2)
    #         for k in range(1, 5):
    #             shaper = ImageEnhance.Sharpness(left_eye)
    #             left_eye = shaper.enhance(1 + k * 0.2)
    #             left_eye.save('{}-{}-{}-100028.jpg'.format(i, j, k))
    pass


def corp_eye(file_path, landmark72):
    image = Image.open(file_path)
    left_region = eye_region(landmark72['left_eye'])
    right_region = eye_region(landmark72['right_eye'])
    left_eye = image.crop(left_region)
    right_eye = image.crop(right_region)

    return left_eye, right_eye


def one_file(file_dir, file_name):
    print('loading... ' + file_name)
    file_path = os.path.join(file_dir, file_name)
    left_save_path = os.path.join(SAVE_DIR, 'left-' + file_name)
    right_save_path = os.path.join(SAVE_DIR, 'right-' + file_name)
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    if not os.path.exists(left_save_path):
        landmark72 = baidu_service(file_path)
        left_eye, right_eye = corp_eye(file_path, landmark72)
        left_eye.resize([SAMPLE_SIZE, SAMPLE_SIZE]).save(left_save_path)
        right_eye.resize([SAMPLE_SIZE, SAMPLE_SIZE]).save(right_save_path)


def one_dir(file_dir):
    file_list = os.listdir(file_dir)
    print(len(file_list))
    for file_name in tqdm(file_list, ncols=70, leave=False, unit='b'):
        one_file(file_dir, file_name)


if __name__ == '__main__':
    # file_name = '100028.jpg'
    # one_file(FILE_DIR, file_name)
    one_dir(FILE_DIR)
    # edges1 = feature.canny(im)
    # edges2 = feature.canny(im, sigma=3)
