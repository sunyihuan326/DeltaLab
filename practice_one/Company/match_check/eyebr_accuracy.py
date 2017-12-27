# coding:utf-8
'''
Created on 2017/12/27.

@author: chk01
'''
from practice_one.Company.match_check.utils import *
from PIL import Image, ImageDraw
import scipy.io as scio
import shutil

org = 'left_eyebrow'

root_dir = 'C:/Users/chk01/Desktop/Delta/image/check/cartoon/' + org

save_dir = '../load_material/feature_mat/' + org


def get_landmark72(full_path):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark"
    }
    result = client.detect(get_file_content(full_path), options=options)
    landmark72 = landmark72_trans(result['result'][0]['landmark72'])
    return landmark72, result['result'][0]['rotation_angle']


def p2f(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    wid = max(x) - min(x)
    hei = max(y) - min(y)
    center = (points[0] + points[4]) / 2
    feature = (points - center) / np.array([wid, hei])
    # feature = points - center
    return feature


def compare_feature(feature):
    features = scio.loadmat(save_dir)
    target = features['data'][:] - feature
    top_index = np.argmin(np.linalg.norm(target, axis=(1, 2)))
    score = round(np.linalg.norm(target, axis=(1, 2))[top_index], 2)
    return top_index, score


def main(file):
    im = Image.open(file).convert("RGB")
    try:
        landmark72, angle = get_landmark72(file)
        if -1 < angle < 1:
            pass
        else:
            im = im.rotate(angle, expand=1)
            print('rotate', file)
            # im.show()
            im.save(file)
            im = Image.open(file)
            landmark72, angle = get_landmark72(file)

        drawSurface = ImageDraw.Draw(im)

        feature = p2f(landmark72[22:30])
        cid, score = compare_feature(feature)

        oldname = root_dir + '/{}.jpg'.format(cid + 1)
        shutil.copyfile(oldname, file.replace('.png', '-src.jpg'))

        landmark72 = tuple(tuple(t) for t in landmark72)

        if not os.path.exists(file.replace('.png', '-res.jpg')):
            wid = im.size[0] // 184
            drawSurface.line(landmark72[22:30], fill=255, width=wid)
            drawSurface.line([landmark72[22], landmark72[29]], fill=255, width=wid)
            im.save(file.replace('.png', '-res.jpg'))
    except KeyError:
        print(file)


if __name__ == '__main__':
    import os

    test_dir = 'C:/Users/chk01/Desktop/eyebr_accuracy'

    for file in os.listdir(test_dir):
        if file.endswith('png'):
            print(file, 'start')
            main(test_dir + '/' + file)
            print(file, 'end')
            print('----------------')
