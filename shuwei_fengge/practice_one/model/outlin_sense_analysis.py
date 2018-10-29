# coding:utf-8
'''
Created on 2017/12/8

@author: sunyihuan
'''

import pandas as pd
import tensorflow as tf
import numpy as np
import scipy.io as sio
from practice_one.model.utils import *
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, accuracy_score

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

dnn_over_out = sio.loadmat('dnn_res_outline13_undersample.mat')
dnn_over_sen = sio.loadmat('dnn_res_sense_undersample.mat')

dnn_res_out = dnn_over_out["ztr"]
dnn_res_sense = dnn_over_sen["ztr"]

X_train, X_test, Y_train, Y_test = load_data('face_1_channel_XY_Points72.mat')
print(X_train.shape, Y_train.shape, Y_test.shape)
Y_train = np.argmax(Y_train, 1)
dnn_res = []
for i in range(dnn_res_out.shape[0]):
    c = 3 * dnn_res_out[i] + dnn_res_sense[i]
    dnn_res.append(c)
dnn_res = list(dnn_res[0])
for i in range(9):
    print(str(i) + '的比例', round(100.0 * list(dnn_res).count(i) / len(dnn_res), 2), '%')
print("^^^^^^^^^^^^")
# print(dnn_res)
for i in range(9):
    print(str(i) + '的比例', round(100.0 * list(Y_train).count(i) / len(Y_train), 2), '%')

print(len(dnn_res), len(Y_train))
k = 0.
ka = 0.
for i in range(len(dnn_res)):
    if dnn_res[i] in accept_ans[Y_train[i]]:
        k += 1. / len(dnn_res)
    if dnn_res[i] in absolute_error[Y_train[i]]:
        ka += 1. / len(dnn_res)
print(k)
print(ka)

# Y_test = np.argmax(Y_test, 1)
