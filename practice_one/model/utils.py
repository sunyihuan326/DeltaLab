# coding:utf-8
'''
Created on 2017/12/9.

@author: chk01
'''
from sklearn.model_selection import train_test_split
import scipy.io as scio


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
