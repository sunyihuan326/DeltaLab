# coding:utf-8
'''
Created on 2017/12/29.

@author: chk01
'''
from practice_one.Company.load_material.utils import *
from practice_one.model.utils import *
from imblearn.over_sampling import RandomOverSampler


def preprocessing(trX, teX, trY, teY):
    res = RandomOverSampler(ratio={0: 700, 1: 700, 2: 700})
    m, w, h = trX.shape
    trX = np.reshape(trX, [m, -1])
    # print(trX.shape, np.squeeze(trY).shape)
    trX, trY = res.fit_sample(trX, np.squeeze(trY))
    new_m = trX.shape[0]
    trX = trX.reshape(new_m, w, h)
    trY = trY.reshape([-1, 1])
    return trX, teX, trY, teY


if __name__ == '__main__':
    file = 'knn-outline.mat'
    X_train, X_test, Y_train, Y_test = load_data(file)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    X_train, X_test, Y_train, Y_test = preprocessing(X_train, X_test, Y_train, Y_test)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    test_num = Y_test.shape[0]
    outline = NearestNeighbor()
    outline.train(X_train, Y_train)
    # distinces = np.zeros([376, ])
    error = 0
    Error = [[2], [], [0]]
    k = 100
    for i in range(376):
        distinces = np.linalg.norm(outline.trX - X_test[i], axis=(1, 2))
        preY = Y_train[np.argsort(distinces)[:k]]

        preY = np.sum(preY) / k
        if int(np.squeeze(preY)) in Error[int(Y_test[i])]:
            error += 1
    print(error / test_num)
