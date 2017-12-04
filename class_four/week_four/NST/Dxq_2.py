# coding:utf-8
'''
Created on 2017/12/1.

@author: chk01
'''
from class_four.week_four.NST.nst_utils import *
import tensorflow as tf
import scipy

STYLE_LAYERS = [
    ('conv1_1', 0.0),
    ('conv2_1', 0.1),
    ('conv3_1', 0.1),
    ('conv4_1', 0.3),
    ('conv5_1', 0.5)]


def gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))


def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.reshape(tf.transpose(a_S), shape=[n_C, -1])
    a_G = tf.reshape(tf.transpose(a_G), shape=[n_C, -1])
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = 1 / (4 * n_H * n_H * n_W * n_W * n_C * n_C) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer


def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C, shape=[n_C, -1])
    a_G_unrolled = tf.reshape(a_G, shape=[n_C, -1])
    J_content = 1 / (4 * n_C * n_H * n_W) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content


def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    return alpha * J_content + beta * J_style


tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()
content_image = scipy.misc.imread("images/1.jpg")
content_image = reshape_and_normalize_image(content_image)
style_image = scipy.misc.imread("images/starry_night.jpg")
style_image = reshape_and_normalize_image(style_image)
generated_image = generate_noise_image(content_image)

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
sess.run(model['input'].assign(content_image))

out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)

print(model['input'])
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)
J = total_cost(J_content, J_style)
# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(0.5)

# define train_step (1 line)
train_step = optimizer.minimize(J)


def model_nn(sess, input_image, num_iterations=200):
    tf.global_variables_initializer().run()
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):

        sess.run(train_step)

        generated_image = sess.run(model['input'])

        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            save_image("output/" + str(i) + ".png", generated_image)

    save_image('output/generated_image.jpg', generated_image)

    return generated_image


model_nn(sess, generated_image, num_iterations=3000)
