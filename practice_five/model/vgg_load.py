# coding:utf-8
'''
Created on 2018/1/15.

@author: chk01
'''
from practice_five.model.nst_utils import *
import scipy.io as scio

model = load_vgg_model("F:/AImodel/imagenet-vgg-verydeep-19.mat")

# X = tf.placeholder(dtype=tf.float32, shape=(None, 64 * 64 * 3))
X = model['input']
Y = tf.placeholder(dtype=tf.float32, shape=(None, 18))

global_step = tf.Variable(0, trainable=False)

vgg_layer = model['conv5_4']
convZ = tf.contrib.layers.flatten(vgg_layer)
ZL = tf.layers.dense(convZ, 18, activation=None, name='output')

learning_rate = tf.train.exponential_decay(0.5,
                                           global_step=global_step,
                                           decay_steps=100, decay_rate=0.9)
learning_rate = tf.maximum(learning_rate, .001)
loss = tf.reduce_mean(tf.square(Y - ZL))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

add_global = global_step.assign_add(1)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

file = '../data/face_top_9.mat'
data = scio.loadmat(file)
trX = data['X'] / 255.
trY = data['Y']
m, features = trY.shape
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(200):
        minibatch_cost = 0.
        num_minibatches = int(m // 64)
        for minibatch_X, minibatch_Y in minibatches(trX, trY, 64, shuffle=True):
            __, _loss, _, res, llr = sess.run([add_global, loss, train_op, ZL, learning_rate],
                                              feed_dict={X: minibatch_X.reshape(-1, 64, 64, 3), Y: minibatch_Y})
            minibatch_cost += _loss / num_minibatches

        if epoch % 10 == 0:
            print('epoch', epoch, 'loss', minibatch_cost)
            print(llr)

    saver.save(sess, "save/model.ckpt")
