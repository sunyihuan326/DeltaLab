# coding:utf-8
'''
Created on 2017/10/31.

@author: chk01
'''

# 读取数据
# 数据预处理-reshape-标准化
# 每一步迭代步骤
# 循环迭代步骤
import numpy as np
from class_one.week_two.lr_utils import load_dataset
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import pylab


def load_data():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
    # print(train_set_x_orig.shape)  # (209, 64, 64, 3)
    # print(train_set_y_orig.shape)  # (1, 209)
    # print(test_set_x_orig.shape)  # (50, 64, 64, 3)
    # print(test_set_y_orig.shape)  # (1, 50)
    # print(classes)  # [b'non-cat' b'cat']
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def activation(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1), np.float32)
    b = 0.0
    assert w.shape == (dim, 1)
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def preprocessing(X):
    pre_set_x = X.reshape(X.shape[0], -1).T
    pre_set_x = pre_set_x / 255.
    return pre_set_x


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = activation(np.dot(w.T, X) + b)
    assert A.shape == (1, m)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}
    return grads, cost


def optimize(w, b, X, Y, epochs, learning_rate, print_cost=False):
    costs = []
    for epoch in range(epochs):
        grads, cost = propagate(w, b, X, Y)
        w = w - learning_rate * grads['dw']
        b = b - learning_rate * grads['db']
        # 保存起来作图可以写进文件
        if epoch % 100 == 0:
            costs.append(cost)

        if print_cost and epoch % 100 == 0:
            print("Cost after iteration %i: %f" % (epoch, cost))
    params = {"w": w,
              "b": b}

    return params, grads, costs


def predict(w, b, X):
    assert w.shape == (X.shape[0], 1)
    pre_Y = activation(np.dot(w.T, X) + b)
    return np.array(pre_Y > 0.5, dtype=int)


def model(X_train, Y_train, X_test, Y_test, epochs=2000, learning_rate=0.5, print_cost=False):
    init_w, init_b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(init_w, init_b, X_train, Y_train, epochs, learning_rate, print_cost)

    pre_train_y = predict(params['w'], params['b'], X_train)
    pre_test_y = predict(params['w'], params['b'], X_test)
    assert pre_train_y.shape == Y_train.shape
    assert pre_test_y.shape == Y_test.shape
    print("train accuracy: {} %".format(100 - np.mean(np.abs(pre_train_y - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(pre_test_y - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": pre_test_y,
         "Y_prediction_train": pre_train_y,
         "w": params['w'],
         "b": params['b'],
         "learning_rate": learning_rate,
         "epochs": epochs}
    return d


def cost_fig(costs, learning_rate):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    # X轴是如何定义？
    return True


def model_compare():
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, epochs=1500, learning_rate=i,
                               print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('epochs')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()


def model_test(w, b):
    my_image = "cat_in_iran.jpg"

    fname = "images/" + my_image
    # ndimage.imread~=misc.imread
    image = np.array(misc.imread(fname, flatten=False))
    my_image = misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T
    my_predicted_image = predict(w, b, my_image)

    plt.imshow(image)
    pylab.show()
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")


if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()
    train_set_x = preprocessing(train_set_x_orig)
    test_set_x = preprocessing(test_set_x_orig)
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, epochs=2000, learning_rate=0.005, print_cost=True)
    # cost_fig(d['costs'], d['learning_rate'])
    # model_compare()
    model_test(d['w'], d['b'])
