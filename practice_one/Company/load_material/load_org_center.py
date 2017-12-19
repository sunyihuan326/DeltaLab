# coding:utf-8
'''
Created on 2017/12/15.

@author: chk01
'''
import requests
import scipy.io as scio


def main():
    resList = requests.get('http://xiaomei.meiyezhushou.com/api/m/sample/feature/all').json()

    left_eye = [[0, 0]] * 32
    right_eye = [[0, 0]] * 32
    left_eyebrow = [[0, 0]] * 25
    right_eyebrow = [[0, 0]] * 25
    # chin = [[0, 0]] * 25
    A_shape = [[0, 0, 0, 0]] * 5
    B_shape = [[0, 0, 0, 0]] * 5
    C_shape = [[0, 0, 0, 0]] * 5
    D_shape = [[0, 0, 0, 0]] * 5
    E_shape = [[0, 0, 0, 0]] * 5
    nose = [[0, 0]] * 12
    lip = [[0, 0]] * 20
    for res in resList['feature_list']:
        x = res['x']
        y = res['y']
        typ = res['type']
        _id = res['title']
        if typ == 'lianxing':
            [typ, wid, hei, _id] = str(_id).split('-')
            # chin[int(_id) - 1] = [x, y]
            if typ == 'A':
                A_shape[int(_id) - 1] = [int(wid), int(hei), x, y]
            elif typ == 'B':
                B_shape[int(_id) - 1] = [int(wid), int(hei), x, y]
            elif typ == 'C':
                C_shape[int(_id) - 1] = [int(wid), int(hei), x, y]
            elif typ == 'D':
                D_shape[int(_id) - 1] = [int(wid), int(hei), x, y]
            else:
                E_shape[int(_id) - 1] = [int(wid), int(hei), x, y]
        elif typ == 'zuiba':
            lip[int(_id) - 1] = [x, y]
        elif typ == 'yanjing':
            [tt, _id] = str(_id).split('-')
            if tt == 'l':
                left_eye[int(_id) - 1] = [x, y]
            else:
                right_eye[int(_id) - 1] = [x, y]
        elif typ == 'bizi':
            nose[int(_id) - 1] = [x, y]
        elif typ == 'meimao':
            [tt, _id] = str(_id).split('-')
            if tt == 'l':
                left_eyebrow[int(_id) - 1] = [x, y]
            else:
                right_eyebrow[int(_id) - 1] = [x, y]
        else:
            pass
    scio.savemat("CartoonPoint", {
        'left_eye': left_eye,
        'right_eye': right_eye,
        'left_eyebrow': left_eyebrow,
        'right_eyebrow': right_eyebrow,
        # 'chin': chin,
        'A_shape': A_shape,
        'B_shape': B_shape,
        'C_shape': C_shape,
        'D_shape': D_shape,
        'E_shape': E_shape,
        'nose': nose,
        'lip': lip
    })


if __name__ == '__main__':
    main()
