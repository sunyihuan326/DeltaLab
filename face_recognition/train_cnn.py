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

# 图片的大小采集的64*64
size = 64
imgs = []
labs = []  # 定义两个数组用来存放图片和标签


# 这里是得到图片的大小，进行统一处理，将图片改成统一大小
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
            # 可能是为了实现像padding的作用
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))

            # 将对应的图片和标签存进数组里面
            imgs.append(img)
            labs.append(path)


readData(my_faces_path)
readData(other_faces_path)

# 将图片数组与标签转换成数组，并给图片做上标签
imgs = np.array(imgs)
labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs])

# 随机划分测试集和训练集，规定测试集的大小，这里是可以自己调的
train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0, 100))

# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

# 将数据转换为小于1的数
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

# 输出一下获取的训练图片和测试图片的长度，也就是大小
print('train size:%s,test size:%s' % (len(train_x), len(test_x)))

# 图片块，每次取100张图片
batch_size = 1
# 计算有多少个batch
num_batch = (len(train_x)) // batch_size

input = tf.placeholder(tf.float32, [None, size, size, 3])
output = tf.placeholder(tf.float32, [None, 2])

# 这里将input在进行处理一下
images = tf.reshape(input, [-1, size, size, 3])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)


# 下面开始进行卷积层的处理
# 第一层卷积，首先输入的图片大小是64*64
def cnnlayer():
    conv1 = tf.layers.conv2d(inputs=images,
                             filters=32,
                             kernel_size=[5, 5],
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu)  # 输出大小是(64*64*32)
    # 第一层池化
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)  # 输出大小是(32*32*32)

    # 第二层卷积
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=32,
                             kernel_size=[5, 5],
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu)  # 输出大小是(32*32*32)

    # 第二层池化
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)  # 输出大小是(16*16*32)

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
                                    strides=2)  # 输出大小是(8*8*32)

    # 第四层卷积
    conv4 = tf.layers.conv2d(inputs=pool3,
                             filters=64,
                             kernel_size=[5, 5],
                             strides=1,
                             padding='same',
                             activation=tf.nn.relu)  # 输出大小是(变成8*8*64）

    # pool3 = tf.layers.max_pooling2d(inputs=conv4,
    #                                 pool_size=[2,2],
    #                                 strides=2)#输出大小是(变成4*4*64)

    # 卷积网络在计算每一层的网络个数的时候要细心一些，不然容易出错
    # 要注意下一层的输入是上一层的输出
    # 平坦化
    flat = tf.reshape(conv4, [-1, 8 * 8 * 64])

    # 经过全连接层
    dense = tf.layers.dense(inputs=flat,
                            units=4096,
                            activation=tf.nn.relu)

    # drop_out处理
    drop_out = tf.layers.dropout(inputs=dense, rate=0.5)

    # 输出层
    logits = tf.layers.dense(drop_out, units=2)
    return logits
    # yield logits


def cnntrain():
    logits = cnnlayer()
    # logits = next(cnnlayer())

    # 交叉熵损失函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=output))
    # 将训练优化方法改成GradientDescentOptimizer发现并没有加快收敛所以又改回AdamOptimizer
    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(output, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    # 合并所有的Op为一个Op
    merged_summary_op = tf.summary.merge_all()

    # 数据保存器的初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 把summary  Op返回的数据写到磁盘里
        summary_writer = tf.summary.FileWriter(
            '/Users/sunyihuan/PycharmProjects/myself/DeltaLab/face_recognition/model_ckpt/tmp',
            graph=tf.get_default_graph())

        for n in range(10):
            # 每次取100(batch_size)张图片
            for i in range(num_batch):
                batch_x = train_x[i * batch_size: (i + 1) * batch_size]
                batch_y = train_y[i * batch_size: (i + 1) * batch_size]

                # 开始训练数据，同时训练三个变量，返回三个数据，
                _, loss, summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                            feed_dict={input: batch_x, output: batch_y})
                summary_writer.add_summary(summary, n * num_batch + i)
                # 打印损失
                print(n * num_batch + i, loss)

                if (n * num_batch + i) % 20 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({input: test_x, output: test_y, keep_prob_5: 1.0, keep_prob_75: 1.0})
                    print("第%f个batch，准确率%f" % (n * num_batch + i, acc))

                    # 准确率大于0.98时保存并退出
                    if acc > 0.9 and n > 2:
                        saver.save(sess,
                                   '/Users/sunyihuan/PycharmProjects/myself/DeltaLab/face_recognition/model_ckpt/train_faces.model',
                                   global_step=n * num_batch + i)
                        sys.exit(0)
        print('accuracy less 0.6, exited!')


cnntrain()
