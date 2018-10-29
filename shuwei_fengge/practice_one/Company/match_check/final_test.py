# coding:utf-8
'''
Created on 2018/1/2.

@author: chk01
'''
from practice_one.Company.load_material.utils import *
import matplotlib.pyplot as plt

root_dir = 'C:/Users/chk01/Desktop/Delta/image'
matrix = 'eye'

org_data = scio.loadmat('../load_material/feature_matrix/{}'.format(matrix))
org_ob = NearestNeighbor()
org_ob.train(X=org_data['X'], Y=org_data['Y'])


def compare_feature(feature):
    org_id = org_ob.predict(feature)
    return org_id


def draw_eye_organ():
    fig = plt.figure()

    org = 'left_eye'
    dir_path = os.listdir(root_dir + '/src/' + org)
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0])
        point_dir = 'final_test_data/eye/eye-' + str(_id)
        ax = fig.add_subplot(5, 5, i + 1)
        ax.set_aspect(1)
        ax.axis('off')

        if not os.path.exists(point_dir + '.mat'):
            full_path = root_dir + '/src/' + org + '/' + sourceDir
            landmark72, _, _, _, _ = get_baseInfo(full_path)
            landmark72 = landmark72_trans(landmark72)
            feature = landmark72[22:30]
            scio.savemat(point_dir, {'Points': feature})
        else:
            feature = scio.loadmat(point_dir)['Points']

        scio.savemat(point_dir, {'Points': feature})
        data_x = np.append(feature[:-1, 0], feature[0, 0])
        data_y = np.append(feature[:-1, 1], feature[0, 1])
        ax.plot(data_x, -data_y)

        ax.scatter(feature[-1][0], -feature[-1][1])
        ax.text(feature[-1][0] - 10, -feature[-1][1] + 20, str(_id))

    plt.savefig('all_eye.jpg')


def draw_eyebr_organ():
    fig = plt.figure()

    org = 'left_eyebrow'
    dir_path = os.listdir(root_dir + '/src/' + org)
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0])
        point_dir = 'final_test_data/eyebr/eyebr-' + str(_id)
        ax = fig.add_subplot(5, 5, i + 1)
        ax.set_aspect(1)
        ax.axis('off')
        if not os.path.exists(point_dir + '.mat'):
            full_path = root_dir + '/src/' + org + '/' + sourceDir
            landmark72, _, _, _, _ = get_baseInfo(full_path)
            landmark72 = landmark72_trans(landmark72)
            feature = landmark72[22:30]
            scio.savemat(point_dir, {'Points': feature})
        else:
            feature = scio.loadmat(point_dir)['Points']

        data_x = np.append(feature[:, 0], feature[0, 0])
        data_y = np.append(feature[:, 1], feature[0, 1])
        ax.plot(data_x, -data_y)

        ax.text(feature[-1][0] - 10, -feature[-1][1] + 20, str(_id))

    plt.savefig('all_eyebr.jpg')


def draw_lip_organ():
    fig = plt.figure()

    org = 'lip'
    dir_path = os.listdir(root_dir + '/src/' + org)
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.split('.')[0])
        point_dir = 'final_test_data/lip/lip-' + str(_id)
        ax = fig.add_subplot(5, 5, i + 1)
        ax.set_aspect(1)
        ax.axis('off')
        if not os.path.exists(point_dir + '.mat'):
            full_path = root_dir + '/src/' + org + '/' + sourceDir
            landmark72, _, _, _, _ = get_baseInfo(full_path)
            landmark72 = landmark72_trans(landmark72)
            feature = landmark72[22:30]
            scio.savemat(point_dir, {'Points': feature})
        else:
            feature = scio.loadmat(point_dir)['Points']

        data_x = np.append(feature[:, 0], feature[0, 0])
        data_y = np.append(feature[:, 1], feature[0, 1])
        ax.plot(data_x, -data_y)

        ax.text(feature[-1][0] - 10, -feature[-1][1] + 20, str(_id))

    plt.savefig('all_eyebr.jpg')


def draw_match_feature(file):
    fig = plt.figure()
    ax1 = fig.add_subplot(652)
    ax1.set_aspect(1)
    ax1.axis('off')
    ax2 = fig.add_subplot(654)
    ax2.set_aspect(1)
    ax2.axis('off')

    landmark72, _, _, _, _ = get_baseInfo(file)
    landmark72 = landmark72_trans(landmark72)
    feature = landmark72[13:22]
    data_x_test = np.append(feature[:-1, 0], feature[0, 0])
    data_y_test = np.append(feature[:-1, 1], feature[0, 1])

    ax1.plot(data_x_test, -data_y_test)
    ax1.scatter(feature[-1][0], -feature[-1][1])
    ax1.text(feature[-1][0] - 10, -feature[-1][1] + 20, file)

    _id = compare_feature(point2feature_eye(landmark72))
    point_dir = 'final_test_data/eye/eye-' + str(_id)
    data = scio.loadmat(point_dir)['Points']
    data_x_match = np.append(data[:-1, 0], data[0, 0])
    data_y_match = np.append(data[:-1, 1], data[0, 1])

    ax2.plot(data_x_match, -data_y_match)
    ax2.scatter(data[-1][0], -data[-1][1])
    ax2.text(data[-1][0] - 10, -data[-1][1] + 20, str(_id))

    dir_path = os.listdir('final_test_data/eye')
    for i, sourceDir in enumerate(dir_path):
        _id = int(sourceDir.replace('.mat', '').split('-')[1])
        if _id not in []:
            point_dir = 'final_test_data/eye/eye-' + str(_id)
            ax = fig.add_subplot(6, 5, 6 + i)
            ax.set_aspect(1)
            ax.axis('off')

            feature = scio.loadmat(point_dir)['Points']

            data_x = np.append(feature[:-1, 0], feature[0, 0])
            data_y = np.append(feature[:-1, 1], feature[0, 1])
            ax.plot(data_x, -data_y)

            ax.scatter(feature[-1][0], -feature[-1][1])
            ax.text(feature[-1][0] - 10, -feature[-1][1] + 20, str(_id))
    plt.savefig(file.replace('.jpg', 'res.jpg'))
    plt.show()


if __name__ == '__main__':
    draw_eye_organ()
    # draw_eyebr_organ()
    # draw_match_feature('6.jpg')
