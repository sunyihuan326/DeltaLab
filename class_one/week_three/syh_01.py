# coding:utf-8
'''
Created on 2017/11/6

@author: sunyihuan
'''
import numpy as np
import matplotlib.pyplot as plt
from class_one.week_three import testCases
import sklearn
import sklearn.datasets
import sklearn.linear_model
import class_one.week_three.testCases
from class_one.week_three.planar_utils import plot_decision_boundary, load_extra_datasets, load_planar_dataset, sigmoid

np.random.seed(1)
# 数据导入
X, Y = load_planar_dataset()
m_x = np.shape(X)
m_y = np.shape(Y)
# print(m_x, m_y)
Y = Y.ravel()


# 应用logistic模型
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, Y.T)
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")


# 计算logistic模型误差
# LR_predictions = clf.predict(X.T)
# print(float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100))

# 定义层数
def layer_sizes(X, Y, n_h=4):
    n_x = X[0]
    n_y = Y[0]
    return (n_x, n_h, n_y)


# X_assess, Y_assess = testCases.layer_sizes_test_case()
# (n_x, n_h, n_y) = layer_sizes(X_assess.shape, Y_assess.shape)
# print(str(n_x), str(n_h), str(n_y))

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert W1.shape == (n_h, n_x)
    assert b1.shape == (n_h, 1)
    assert W2.shape == (n_y, n_h)
    assert b2.shape == (n_y, 1)

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


# 测试initialize_parameters函数
# n_x, n_h, n_y = testCases.initialize_parameters_test_case()
#
# parameters = initialize_parameters(n_x, n_h, n_y)
# print(str(parameters["W1"]),str(parameters["b2"]))

# 定义forward propagation
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert A2.shape == (1, X.shape[1])
    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2

    }
    return A2, cache


# 测试forward_propagation函数
# X_assess, parameters = testCases.forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# print(np.mean(cache["Z2"]))

# 计算成本函数
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = - 1 / m * np.sum(logprobs)  # 求和

    cost = np.squeeze(cost)
    assert isinstance(cost, float)
    return cost


# 测试compute_cost函数
# A2, Y_assess, parameters = testCases.compute_cost_test_case()
# print(str(compute_cost(A2, Y_assess, parameters)))

# 定义backward propagation
def backward_propagation(X, Y, parameters, cache):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads


# 测试backward_propagation函数
# parameters, cache, X_assess, Y_assess = testCases.backward_propagation_test_case()
# grads = backward_propagation(X_assess, Y_assess, parameters, cache)
# print(str(grads["dW2"]), str(grads["db2"]))


def update_parameters(parameters, grads, learning_rate=1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - db1 * learning_rate
    W2 = W2 - learning_rate * dW2
    b2 = b2 - db2 * learning_rate

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


# 测试update_parameters函数
# parameters, grads = testCases.update_parameters_test_case()
# parameters = update_parameters(parameters, grads)
# print(str(parameters["b1"]), str(parameters["b2"]))

def nn_model(X, Y, n_h, num_iterations=1000, print_cost=False):
    np.random.seed(3)
    n_x, n_h, n_y = layer_sizes(X.shape, Y.shape, n_h)
    parameters = initialize_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(X, Y, parameters, cache)
        parameters = update_parameters(parameters, grads, learning_rate=1.2)

        if print_cost and i % 1000 == 0:
            print("Cost after interation %i:%f" % (i, cost))
    return parameters


# 测试nn_model函数
# X_assess, Y_assess = testCases.nn_model_test_case()
#
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
# print(str(parameters["W1"]))


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X, parameters)
    predictions = np.array(A2 > 0.5)
    ### END CODE HERE ###

    return predictions

# 测试predict函数
# parameters, X_assess = testCases.predict_test_case()
#
# predictions = predict(parameters, X_assess)
# print("predictions mean = " + str(np.mean(predictions)))
