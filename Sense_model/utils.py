# coding:utf-8 
'''
created on 2018/3/8

@author:sunyihuan
'''

from sklearn.model_selection import train_test_split
import scipy.io as scio
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler


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
                                                        shuffle=True)

    return X_train, X_test, Y_train, Y_test


def preprocessing(trX, teX, trY, teY):
    res = SMOTE(ratio="auto", random_state=42)
    trY = np.argmax(trY, 1)
    #
    trX, trY = res.fit_sample(trX, trY)
    # # teX, teY = res.fit_sample(teX, teY)
    # # teY = np.argmax(teY, 1)
    trY = np.eye(3)[trY]
    return trX, teX, trY, teY
