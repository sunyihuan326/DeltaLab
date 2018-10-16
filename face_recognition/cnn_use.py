# coding:utf-8 
'''
created on 2018/10/13

@author:sunyihuan
'''
import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

my_faces_path = '/Users/sunyihuan/Desktop/like'
other_faces_path = '/Users/sunyihuan/Desktop/unlike'

# 将得到的自己的图片和其他图片进行处理
size = 64
imgs = []
labs = []  # 定义两个数组用来存放图片和标签


def getPaddingSize(img):
    h, w, _ = img.shape  # 获得图片的宽和高还有深度
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left

    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top

    else:
        pass
    return top, bottom, left, right


def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top, bottom, left, right = getPaddingSize(img)

            # 将图片放大，扩充图片边缘部分，这里不是很理解为什么要扩充边缘部分

            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)  # 将对应的图片和标签存进数组里面


readData(my_faces_path)
readData(other_faces_path)

# 将图片数组与标签转换成数组
imgs = np.array(imgs)
labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs])

# 随机划分测试集和训练集
train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0, 100))

# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

# 将数据转换为小于1的数，二值化使处理变得更简单
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

# 获取训练图片和测试图片的长度，也就是大小
print('train size:%s,test size:%s' % (len(train_x), len(test_x)))

# 图片块，每次取100张图片
batch_size = 100
num_batch = (len(train_x)) // batch_size  # 计算总共多少轮

input = tf.placeholder(tf.float32, [None, size, size, 3])
output = tf.placeholder(tf.float32, [None, 2])  # 输出加两个，true or false
# 这里注意的是tf.reshape不是np.reshape
# images = tf.reshape(input,[-1,size,size,3])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


# 下面开始进行卷积层的处理
# 第一层卷积，首先输入的图片大小是64*64
def cnnlayer():
    # 第一层卷积
    conv1 = tf.layers.conv2d(inputs=input,
                             filters=32,
                             kernel_size=[5, 5],
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu)  # (64*64*32)
    # 第一层池化
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)  # (32*32*32)

    # 第二层卷积
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=32,
                             kernel_size=[5, 5],
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu)  # (32*32*32)

    # 第二层池化
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)  # (16*16*32)

    # 第三层卷积
    conv3 = tf.layers.conv2d(inputs=pool2,
                             filters=32,
                             kernel_size=[5, 5],
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu)  # (变成16*16*32)
    # 第三层池化
    pool3 = tf.layers.max_pooling2d(inputs=conv3,
                                    pool_size=[2, 2],
                                    strides=2)  # (8*8*32)

    # 第四层卷积
    conv4 = tf.layers.conv2d(inputs=pool3,
                             filters=64,
                             kernel_size=[5, 5],
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu)  # (变成8*8*64）
    # pool3 = tf.layers.max_pooling2d(inputs=conv4,
    #                                 pool_size=[2,2],
    #                                 strides=2)#(变成4*4*6)

    # 卷积网络在计算每一层的网络个数的时候要细心一些
    # 卷积层加的padding为same是不会改变卷积层的大小的
    # 要注意下一层的输入是上一层的输出
    # 平坦化
    flat = tf.reshape(conv4, [-1, 8 * 8 * 64])

    # 经过全连接层
    dense = tf.layers.dense(inputs=flat,
                            units=4096,
                            activation=tf.nn.relu)

    # drop_out，flat打错一次
    drop_out = tf.layers.dropout(inputs=dense, rate=0.2)

    # 输出层
    logits = tf.layers.dense(drop_out, units=2)
    return logits
    # yield logits


out = cnnlayer()
# out = next(cnnlayer())
predict = tf.argmax(out, 1)
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, tf.train.latest_checkpoint(
    '/Users/sunyihuan/PycharmProjects/myself/DeltaLab/face_recognition/model_ckpt'))


def is_my_face(image):
    out_ = sess.run(out, feed_dict={input: [image / 255.0]})
    res = sess.run(predict, feed_dict={input: [image / 255.0]})
    print(out_)
    print(res)


filename = "/Users/sunyihuan/Desktop/IMG_3881.JPG"
img = cv2.imread(filename)
img = cv2.resize(img, (size, size))
is_my_face(img)
