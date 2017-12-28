# coding:utf-8
'''
Created on 2017/12/27.

@author: chk01
'''
from practice_one.Company.load_material.utils import *
from PIL import Image, ImageDraw
import scipy.io as scio
import shutil

org = 'left_eyebrow'

root_dir = 'C:/Users/chk01/Desktop/Delta/image/check/cartoon/' + org

org_data = scio.loadmat('../load_material/feature_matrix/{}'.format('eyebr'))
org_ob = NearestNeighbor()
org_ob.train(X=org_data['X'], Y=org_data['Y'])


def compare_feature(feature):
    org_id = org_ob.predict(feature)
    return org_id


def main(file):
    im = Image.open(file).convert("RGB")
    try:
        landmark72, angle, _, _, _ = get_baseInfo(file)
        if -1 < angle < 1:
            pass
        else:
            im = im.rotate(angle, expand=1)
            # im.show()
            im.save(file)
            im = Image.open(file)
            landmark72, angle, _, _, _ = get_baseInfo(file)

        landmark72 = landmark72_trans(landmark72)
        drawSurface = ImageDraw.Draw(im)

        feature = point2feature_ebr(landmark72)
        cid = compare_feature(feature)

        oldname = root_dir + '/{}.jpg'.format(cid)
        shutil.copyfile(oldname, file.replace('.png', '-src.jpg'))

        landmark72 = tuple(tuple(t) for t in landmark72)

        if not os.path.exists(file.replace('.png', '-res.jpg')):
            wid = im.size[0] // 184
            drawSurface.line(landmark72[22:30], fill=255, width=wid)
            drawSurface.line([landmark72[22], landmark72[29]], fill=255, width=wid)
            im.save(file.replace('.png', '-res.jpg'))
    except Exception as e:
        print('Error', e)


if __name__ == '__main__':
    import os

    test_dir = 'C:/Users/chk01/Desktop/eyebr_accuracy'
    for file in os.listdir(test_dir):
        if file.endswith('png'):
            print(file, 'start')
            main(test_dir + '/' + file)
            print(file, 'end')
            print('----------------')
