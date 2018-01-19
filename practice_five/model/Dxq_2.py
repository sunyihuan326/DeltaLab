# coding:utf-8
'''
Created on 2017/12/1.

@author: chk01
'''
from practice_five.model.nst_utils import *
import tensorflow as tf
import scipy

import numpy as np

a = np.array([[1, 2, 3, 2], [1, 2, 3, 2]]).reshape(2, -1)
b = np.array([[1, 2, 3, 2], [2, 2, 2, 2]]).reshape(2, -1)
print((a - b).shape)
print(a - b)
c = [2, 1]
d = ((a - b) * (c * 2))
print(d)
assert 1 == 0

tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

model = load_vgg_model("F:/AImodel/imagenet-vgg-verydeep-19.mat")

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
