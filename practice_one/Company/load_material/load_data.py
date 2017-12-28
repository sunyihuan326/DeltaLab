# coding:utf-8
'''
Created on 2017/11/24.

@author: chk01
'''
from skimage import feature as ft
from practice_one.Company.load_material.utils import *


# nose eye
def get_hog_feature():
    feature = {
        'left_eye': '',
        'right_eye': '',
        'nose': '',
    }
    for org in ['nose', 'left_eye', 'right_eye']:
        organ = Image.open("real/{}/{}_{}.jpg".format(org, org, 2)).convert("L")
        feature[org] = ft.hog(organ,  # input image
                              orientations=9,  # number of bins
                              pixels_per_cell=(4, 4),  # pixel per cell
                              cells_per_block=(10, 10),  # cells per blcok
                              block_norm='L2-Hys',  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                              transform_sqrt=False,  # power law compression (also known as gamma correction)
                              feature_vector=True,  # flatten the final vectors
                              visualise=True)  # return HOG map
        print(feature[org][0])
        res = Image.fromarray(feature[org][1] * 200)
        res.show()
    return feature


def get_face_feature():
    for typ in ['A', 'B', 'C', 'D', 'E']:
        print('开始{}型导入'.format(typ))
        dir_path = os.listdir(root_dir + '/src/face_' + typ)
        m = len(dir_path)
        n = 13
        X = np.zeros([m, n, 2])
        Y = np.zeros([m, 1])
        for i, sourceDir in enumerate(dir_path):
            _id = int(sourceDir.split('.')[0]) - 1
            full_path = root_dir + '/src/face_' + typ + '/' + sourceDir
            landmark72, _, _, _, _ = get_baseInfo(full_path)
            landmark72 = landmark72_trans(landmark72)

            _data = point2feature_chin(landmark72)
            X[_id] = _data
            Y[_id] = _id + 1
            print('load--->{}---图{}'.format(typ, _id))
        scio.savemat('feature_matrix/face_' + typ, {"X": X, "Y": Y})
        print('完成{}导入'.format(typ))


if __name__ == '__main__':
    get_face_feature()
    # get_point_feature()
    # data = scio.loadmat('feature_mat/left_eye')
    # print(data['data'].shape)
