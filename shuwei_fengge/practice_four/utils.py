# coding:utf-8
'''
Created on 2017/12/9.

@author: chk01
'''
import math
import scipy.io as scio
import numpy as np

from sklearn.model_selection import train_test_split

accept_ans = [
    [0, 1, 3],
    [1, 0, 2, 4],
    [2, 1, 5],
    [3, 0, 4, 6],
    [4, 1, 3, 5, 7],
    [5, 2, 4, 8],
    [6, 3, 7],
    [7, 6, 4, 8],
    [8, 7, 5],
]
absolute_error = [
    [2, 5, 6, 7, 8],
    [6, 7, 8],
    [0, 3, 6, 7, 8],
    [2, 5, 8],
    [0, 2, 6, 8],
    [0, 3, 6],
    [0, 1, 2, 5, 8],
    [0, 1, 2],
    [0, 1, 2, 3, 6]
]


def load_data(file, test_size=0.25):
    '''
    :param file:the name of dataset
    :param test_size:float, int, None, optional
                If float, should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the test split. If int, represents the
                absolute number of test samples.
    :return: X_train, X_test, Y_train, Y_test shape:[m,features]--[m,classes]
    '''
    data_train = scio.loadmat(file)

    X_train, X_test, Y_train, Y_test = train_test_split(data_train['X'], data_train['Y'], test_size=test_size,
                                                        shuffle=True, random_state=16)

    return X_train, X_test, Y_train, Y_test


def data_check(data):
    '''
    :param data: shape:[m,classes]
    '''
    res = list(np.argmax(data, 1))
    num = len(res)
    classes = data.shape[1]
    for i in range(classes):
        tt = 0
        for j in accept_ans[i]:
            tt += res.count(j)
        print(str(i) + '的比例', round(100.0 * res.count(i) / num, 2), '%', '全选' + str(i) + '的不出错比例',
              round(100.0 * tt / num, 2), '%')
    print('<------------------分割线---------------------->')


def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches


def minibatches(X, Y, batch_size=64, shuffle=True):
    assert len(X) == len(Y)
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
    for start_idx in range(0, len(X) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield X[excerpt], Y[excerpt]


def accuracy_cal(res_matrix, typ):
    num = np.sum(res_matrix)
    print('----------------', str(typ).upper(), '------------------')
    print(0, '准确率 {}%'.format(round(100 * res_matrix[0, 0] / np.sum(res_matrix[:, 0]), 2)),
          '| 召回率 {}%'.format(round(100 * res_matrix[0, 0] / np.sum(res_matrix[0, :]), 2)),
          '| 错误率 {}%'.format(round(100 * res_matrix[0, 2] / np.sum(res_matrix[0, :]), 2)),
          '| 出现率 {}%'.format(round(100 * np.sum(res_matrix[:, 0]) / num, 2)))

    print(2, '准确率 {}%'.format(round(100 * res_matrix[2, 2] / np.sum(res_matrix[:, 2]), 2)),
          '| 召回率 {}%'.format(round(100 * res_matrix[2, 2] / np.sum(res_matrix[2, :]), 2)),
          '| 错误率 {}%'.format(round(100 * res_matrix[2, 0] / np.sum(res_matrix[2, :]), 2)),
          '| 出现率 {}%'.format(round(100 * np.sum(res_matrix[:, 2]) / num, 2)))

    print(1, '准确率 {}%'.format(round(100 * res_matrix[1, 1] / np.sum(res_matrix[:, 1]), 2)),
          '| 召回率 {}%'.format(round(100 * res_matrix[1, 1] / np.sum(res_matrix[1, :]), 2)),
          '| 识别为0率 {}%'.format(round(100 * res_matrix[1, 0] / np.sum(res_matrix[1, :]), 2)),
          '| 识别为2率 {}%'.format(round(100 * res_matrix[1, 2] / np.sum(res_matrix[1, :])), 2),
          '| 出现率 {}%'.format(round(100 * np.sum(res_matrix[:, 1]) / num, 2)))

    print('准确率 {}%'.format(round(100 * (res_matrix[0, 0] + res_matrix[1, 1] + res_matrix[2, 2]) / num, 2)))
    print('单维度可接受率 {}%'.format(
        round(100 - 100 * res_matrix[0, 2] / num - 100 * res_matrix[2, 0] / num, 2)))
    print('单维度最优可接受率 {}%'.format(
        round(100 - 100 * res_matrix[0, 2] / num - 100 * res_matrix[2, 0] / num - 100 * res_matrix[1, 0] / num - 100 *
              res_matrix[1, 2] / num, 2)))
    print('------------------------------------------------------')
