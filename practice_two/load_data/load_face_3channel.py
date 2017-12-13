# coding:utf-8
'''
Created on 2017/12/2.

@author: chk01
'''
from practice_two.load_data.utils import *
from PIL import Image
from io import BytesIO
import os


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
    data = np.zeros(shape=(num, 64 * 64 * 3))
    data_Y = np.zeros([num, 9])
    for i, res in enumerate(resList):
        point_dir = '../data/image3channel/Image-Point-{}.mat'.format(res['_id'])
        label_dir = '../data/label/Image-Label-{}.mat'.format(res['_id'])
        if os.path.exists(point_dir):
            points = scio.loadmat(point_dir)['Points']
        else:
            print('read_{}_data------->loading----->start'.format(i))
            points = get_landmark72(res['face_img'], 'url')
            scio.savemat(point_dir, {'Points': points})

        response = requests.get(res['face_img'])
        image = Image.open(BytesIO(response.content)).convert("RGB")
        region = get_face_box(points)

        data[i, :] = np.array(image.crop(region).resize([64, 64])).reshape(1, 64 * 64 * 3)
        data_Y[i, :] = scio.loadmat(label_dir)['Label']
        print('read_{}_data------->loading----->end'.format(i))

    scio.savemat('face_3_channel_XY64', {"X": data, "Y": data_Y})


if __name__ == '__main__':
    main()
    # pass
