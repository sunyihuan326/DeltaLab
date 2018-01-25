# coding:utf-8
'''
Created on 2018/1/22.

@author: chk01
'''
import requests
import scipy.io as scio
import numpy as np
from PIL import Image
from io import BytesIO
import os


def download_url_img(url, fpath, mode='RGBA'):
    fdir = fpath[:fpath.rfind('/')]
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert(mode)
    img.save(fpath)
    return fpath


def download_face_material():
    resList = requests.get('http://devxm.meiyezhushou.com/api/m/sample/face/list').json()
    img_domain = resList['domain']
    ii = 0
    for res in resList['face_list']:
        if res['type'] == 'face':
            ii += 1
            print('loading ' + str(ii))
            update_one_face_material(res, img_domain)
            print('loading over' + str(ii))


def update_one_face_material(face_obj, img_domain):
    title = face_obj['title']
    typ, wid, hei, _id = title.split('-')

    # model_img = img_domain + face_obj['face_img']
    # download_url_img(model_img, 'material/cartoon/face/{}/model/{}-{}.jpg'.format(typ, typ, _id), mode='RGB')
    mark = typ + '-' + _id
    if mark in ['A-1']:
        for i in range(3):
            # 冷中暖
            for j in range(3):
                # 浅中深
                file_name = str(3 * j + i + 1)
                key = '_{}_{}'.format(i, j)
                img = img_domain + face_obj[key]
                download_url_img(img, 'material/cartoon/face/{}/{}/{}-{}.png'.format(typ, file_name, typ, _id))
    return True


def update_position():
    resList = requests.get('http://devxm.meiyezhushou.com/api/m/sample/face/list').json()

    left_eye = np.zeros([32, 2]) + 999
    right_eye = np.zeros([32, 2]) + 999
    left_eyebrow = np.zeros([32, 2]) + 999
    right_eyebrow = np.zeros([32, 2]) + 999
    A_shape = np.zeros([32, 6]) + 999
    B_shape = np.zeros([32, 6]) + 999
    C_shape = np.zeros([32, 6]) + 999
    D_shape = np.zeros([32, 6]) + 999
    E_shape = np.zeros([32, 6]) + 999
    nose = np.zeros([32, 2]) + 999
    lip = np.zeros([32, 3]) + 999

    for res in resList['face_list']:
        x = res['pos_x']
        y = res['pos_y']
        typ = res['type']
        _id = res['title']
        if typ == 'face':
            [typ, wid, hei, _id] = str(_id).split('-')
            lower_x = res['lower_x']
            lower_y = res['lower_y']
            header_x = res['header_x']
            header_y = res['header_y']
            chin_data = [int(wid), int(hei), int(lower_x), int(lower_y), int(header_x), int(header_y)]
            if typ == 'A':
                A_shape[int(_id) - 1] = chin_data
            elif typ == 'B':
                B_shape[int(_id) - 1] = chin_data
            elif typ == 'C':
                C_shape[int(_id) - 1] = chin_data
            elif typ == 'D':
                D_shape[int(_id) - 1] = chin_data
            else:
                E_shape[int(_id) - 1] = chin_data
        elif typ == 'mouth':
            [hou, _id] = str(_id).split('-')
            lip[int(_id) - 1] = [x, y, hou]
        elif typ == 'eye':
            [tt, _id] = str(_id).split('-')
            if tt == 'l':
                left_eye[int(_id) - 1] = [x, y]
            else:
                right_eye[int(_id) - 1] = [x, y]
        elif typ == 'nose':
            nose[int(_id) - 1] = [x, y]
        elif typ == 'brow':
            [tt, _id] = str(_id).split('-')
            if tt == 'l':
                left_eyebrow[int(_id) - 1] = [x, y]
            else:
                right_eyebrow[int(_id) - 1] = [x, y]
        else:
            pass
    fdir = 'material/feature_matrix'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    scio.savemat(fdir + "/CartoonPoint", {
        'left_eye': left_eye,
        'right_eye': right_eye,
        'left_eyebrow': left_eyebrow,
        'right_eyebrow': right_eyebrow,
        'A_shape': A_shape,
        'B_shape': B_shape,
        'C_shape': C_shape,
        'D_shape': D_shape,
        'E_shape': E_shape,
        'nose': nose,
        'lip': lip
    })


def update_one_nose_material(nose_obj, img_domain):
    _id = nose_obj['title']

    model_img = img_domain + nose_obj['face_img']
    download_url_img(model_img, 'material/cartoon/nose/model/{}.jpg'.format(_id), mode='RGB')

    for i in range(3):
        # 冷中暖
        for j in range(3):
            # 浅中深
            file_name = str(3 * j + i + 1)
            key = '_{}_{}'.format(i, j)
            if nose_obj[key]:
                img = img_domain + nose_obj[key]
                download_url_img(img, 'material/cartoon/nose/{}/{}.png'.format(file_name, _id))
    return True


def download_nose_material():
    resList = requests.get('http://devxm.meiyezhushou.com/api/m/sample/face/list').json()
    img_domain = resList['domain']
    ii = 0
    for res in resList['face_list']:
        if res['type'] == 'nose':
            ii += 1
            print('loading ' + str(ii))
            update_one_nose_material(res, img_domain)
            print('loading over ' + str(ii))


def update_one_eye_material(eye_obj, img_domain):
    title = eye_obj['title']
    tt, _id = str(title).split('-')
    if tt == 'l':
        model_img = img_domain + eye_obj['face_img']
        download_url_img(model_img, 'material/cartoon/left_eye/model/{}.jpg'.format(_id), mode='RGB')

    for i in range(3):
        # 冷中暖
        for j in range(3):
            # 浅中深
            file_name = str(3 * j + i + 1)
            key = '_{}_{}'.format(i, j)
            if eye_obj[key]:
                img = img_domain + eye_obj[key]
                rr = 'left' if tt == 'l' else 'right'
                download_url_img(img, 'material/cartoon/{}_eye/{}/{}.png'.format(rr, file_name, _id))
    return True


def download_eye_material():
    resList = requests.get('http://devxm.meiyezhushou.com/api/m/sample/face/list').json()
    img_domain = resList['domain']
    ii = 0
    for res in resList['face_list']:
        if res['type'] == 'eye':
            ii += 1
            print('loading ' + str(ii))
            update_one_eye_material(res, img_domain)
            print('loading over ' + str(ii))


def update_one_brow_material(brow_obj, img_domain):
    title = brow_obj['title']
    tt, _id = str(title).split('-')
    if tt == 'l':
        model_img = img_domain + brow_obj['face_img']
        download_url_img(model_img, 'material/cartoon/left_eyebrow/model/{}.jpg'.format(_id), mode='RGB')

    if brow_obj['img']:
        img = img_domain + brow_obj['img']
        rr = 'left' if tt == 'l' else 'right'
        download_url_img(img, 'material/cartoon/{}_eyebrow/{}.png'.format(rr, _id))


def download_brow_material():
    resList = requests.get('http://devxm.meiyezhushou.com/api/m/sample/face/list').json()
    img_domain = resList['domain']
    ii = 0
    for res in resList['face_list']:
        if res['type'] == 'brow':
            ii += 1
            print('loading ' + str(ii))
            update_one_brow_material(res, img_domain)
            print('loading over ' + str(ii))


def download_lip_material():
    resList = requests.get('http://devxm.meiyezhushou.com/api/m/sample/face/list').json()
    img_domain = resList['domain']
    ii = 0
    for res in resList['face_list']:
        if res['type'] == 'mouth':
            ii += 1
            print('loading ' + str(ii))
            update_one_lip_material(res, img_domain)
            print('loading over ' + str(ii))


def update_one_lip_material(lip_obj, img_domain):
    _id = lip_obj['title']

    model_img = img_domain + lip_obj['face_img']
    download_url_img(model_img, 'material/cartoon/lip/model/{}.jpg'.format(_id), mode='RGB')

    if lip_obj['img']:
        img = img_domain + lip_obj['img']
        download_url_img(img, 'material/cartoon/lip/{}.png'.format(_id))


if __name__ == '__main__':
    update_position()
