# coding:utf-8 
'''
created on 2018/2/3

@author:sunyihuan
'''

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from class_four.week_four.NST.nst_utils import *
import numpy as np
import tensorflow as tf

vgg_parameters_file = "/Users/sunyihuan/Desktop/modelParameters/imagenet-vgg-verydeep-19.mat"
model = load_vgg_model(vgg_parameters_file)
# print(model)

content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image)


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    (m, n_H, n_W, n_C) = a_C.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C, shape=(n_H * n_W, n_C))
    a_G_unrolled = tf.reshape(a_G, shape=(n_H * n_W, n_C))
    J_content = 1.0 / (4 * n_H * n_W * n_C) * \
                tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled))

    return J_content


style_image = scipy.misc.imread("images/monet_800600.jpg")
imshow(style_image)

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, A, transpose_a=False, transpose_b=True)
    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    m, n_H, n_W, n_C = a_S.get_shape().as_list()
    a_S = tf.reshape(tf.transpose(a_S), shape=[n_C, -1])
    a_G = tf.reshape(tf.transpose(a_G), shape=[n_C, -1])
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = 1.0 / (4 * pow(n_C, 2) * pow((n_H * n_W), 2)) * tf.reduce_sum(tf.reduce_sum(tf.square(GG - GS)))
    return J_style_layer

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    return J


tf.reset_default_graph()
sess = tf.InteractiveSession()

content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])

sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)
sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_content)
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)


def model_nn(sess, input_image, num_iterations=200):
    tf.global_variables_initializer()
    sess.run(model["input"].assign(input_image))
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model["input"])
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            save_image("output/" + str(i) + ".png", generated_image)
    save_image('output/generated_image.jpg', generated_image)

    return generated_image

model_nn(sess, generated_image)
