# coding:utf-8
'''
Created on 2018/1/4.

@author: chk01
'''
from sklearn.metrics import confusion_matrix, classification_report
from practice_four.utils import *
from collections import Counter

outline_parameters = scio.loadmat('parameter/outline64x64_parameter-2500.mat')
sense_parameters = scio.loadmat('parameter/sense64_parameter-3500.mat')


def get_outline64(trX):
    W = outline_parameters['W1']
    b = outline_parameters['b1']
    Z = np.add(np.matmul(trX, W.T), b)
    return np.squeeze(np.argmax(Z, 1))


def get_sense64(trX):
    W = sense_parameters['W1']
    b = sense_parameters['b1']
    Z = np.add(np.matmul(trX, W.T), b)
    return np.squeeze(np.argmax(Z, 1))


def main():
    data = scio.loadmat('data/style64x64.mat')
    trX = data['X']
    trY = np.argmax(data['Y'], 1)
    m, _ = trX.shape
    outline = get_outline64(trX)
    sense = get_sense64(trX)
    style = 3 * outline + sense

    train_res_matrix = confusion_matrix(y_true=trY, y_pred=style)
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
    print(train_res_matrix)

    return True


if __name__ == '__main__':
    main()
