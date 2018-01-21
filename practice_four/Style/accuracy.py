# coding:utf-8
'''
Created on 2018/1/4.

@author: chk01
'''
from sklearn.metrics import confusion_matrix, classification_report
from practice_four.utils import *
from collections import Counter
import numpy

outline_parameters = scio.loadmat('best_parameter/outline64x64_parameter-2500.mat')
outline_2classes_parameters = scio.loadmat('best_parameter/outline64x64-2classes_parameter-4000.mat')
sense_parameters = scio.loadmat('best_parameter/sense64_parameter-5500.mat')
sense_2classes_parameters = scio.loadmat('best_parameter/sense64_02_parameter-2000.mat')

LabelToOutline = [0, 0, 0, 1, 1, 1, 2, 2, 2]
LabelToSense = [0, 1, 2, 0, 1, 2, 0, 1, 2]


def get_outline64(trX):
    W = outline_parameters['W1']
    b = outline_parameters['b1']
    Z = np.add(np.matmul(trX, W.T), b)
    return np.squeeze(np.argmax(Z, 1))


def get_outline64_2classes(trX):
    W = outline_2classes_parameters['W1']
    b = outline_2classes_parameters['b1']
    Z = np.add(np.matmul(trX, W.T), b)
    return np.squeeze(np.argmax(Z, 1))


def get_sense64(trX):
    W = sense_parameters['W1']
    b = sense_parameters['b1']
    Z = np.add(np.matmul(trX, W.T), b)
    return np.squeeze(np.argmax(Z, 1))


def get_sense64_2classes(trX):
    W = sense_2classes_parameters['W1']
    b = sense_2classes_parameters['b1']
    Z = np.add(np.matmul(trX, W.T), b)
    return np.squeeze(np.argmax(Z, 1))


def report(y_true, y_pred, typ):
    print('---------------', str(typ).upper(), '----------------')
    res = classification_report(y_true=y_true, y_pred=y_pred)
    print(res)


def analysis(trY, style):
    m = len(trY)
    train_res_matrix = confusion_matrix(y_true=trY, y_pred=style)
    print(train_res_matrix)

    correct = 0
    error = 0
    for i in range(m):
        if style[i] in accept_ans[trY[i]]:
            correct += 1
        elif style[i] in absolute_error[trY[i]]:
            error += 1

    print('准确率：', round(train_res_matrix.trace() / m, 2))
    print('可接受率：', round(correct / m, 2))
    print('原则性错误率：', round(error / m, 2))
    for i in range(9):
        num = np.sum(train_res_matrix[i, :])
        acc = round((train_res_matrix[i, i]) / num, 2)
        accept_num = 0
        err_num = 0
        for j in accept_ans[i]:
            accept_num += train_res_matrix[i, j]
        for k in absolute_error[i]:
            err_num += train_res_matrix[i, k]
        accept = round(accept_num / num, 2)
        err = round(err_num / num, 2)
        print('输入--------', i, '--------------')
        print('准确率：', acc, '|可接受率：', accept, '|原则性错误率：', err)
        print('----------------------------------------------------')
        # c = Counter(style)
        # c.most_common()
        # print(c)
    print("predict result:")
    for i in range(9):
        print(str(i) + "比例", round(100 * list(style).count(i) / len(list(style)), 2), "%")
    print("true result:")
    for i in range(9):
        print(str(i) + "比例", round(100 * list(trY).count(i) / len(list(trY)), 2), "%")



def main():
    file = 'data/style64x64.mat'
    X_train_org, X_test_org, Y_train_org, Y_test_org = load_data(file, test_size=0.2)
    trX = X_test_org
    trY = np.argmax(Y_test_org, 1)
    cor_outline = [LabelToOutline[l] for l in trY]
    cor_sense = [LabelToSense[ll] for ll in trY]

    sense = get_sense64(trX / 255.)
    outline = get_outline64(trX / 255.)

    sense_2 = get_sense64_2classes(trX / 255.)
    outline_2 = get_outline64_2classes(trX / 255.)

    report(cor_outline, outline, 'outline')
    report(cor_sense, sense, 'sense')

    style = 3 * outline + sense

    outline_merge = outline.copy()
    dif_idx = np.flatnonzero((outline - outline_2 * 2) != 0)
    outline_merge[dif_idx] = 1
    report(cor_outline, outline_merge, 'outline-merge')

    sense_merge = sense.copy()
    dif_idx = np.flatnonzero((sense - sense_2 * 2) != 0)
    sense_merge[dif_idx] = 1
    report(cor_sense, sense_merge, 'sense-merge')

    style_2 = 3 * outline_merge + sense_merge

    analysis(trY, style_2)
    return True


if __name__ == '__main__':
    main()
