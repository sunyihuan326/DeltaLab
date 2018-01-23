# coding:utf-8
'''
Created on 2018/1/15.

@author: chk01
'''
from practice_two.load_data.utils import *
from PIL import Image, ImageDraw
import tensorflow as tf
import scipy.io as scio
import matplotlib.pyplot as plt


def get_face_box(points):
    X = points[:, 0]
    Y = points[:, 1]
    min_x = min(X)
    max_x = max(X)
    min_y = min(Y)
    max_y = max(Y)
    wid = max(max_y - min_y, max_x - min_x)
    wid = 1.8 * wid
    new_x = min_x - (wid - (max_x - min_x)) // 2
    new_y = min_y - (wid - (max_y - min_y)) // 2
    p = 0.2
    region = [new_x, new_y - p * wid, new_x + wid, new_y + (1 - p) * wid]
    return region, wid


def main():
    img_path = '44.jpg'
    image = Image.open(img_path).convert("L")

    points = get_landmark72(img_path)
    region, width = get_face_box(points)

    new_x = region[0]
    new_y = region[1]
    res = np.array(image.crop(region).resize([64, 64]))
    tt = np.squeeze(predict(res)).reshape(-1, 2) * width / 64 + [new_x, new_y]
    plt.scatter(points[:, 0], -points[:, 1])
    plt.scatter(tt[:, 0], -tt[:, 1])
    plt.axis('equal')
    plt.show()

    drawSurface = ImageDraw.Draw(image)
    landmark72 = tuple(tuple(t) for t in tt)
    rr = tuple(tuple(t) for t in points)
    drawSurface.line(rr[:13], fill=255, width=5)
    # drawSurface.polygon([landmark72[2:5],landmark72[-3]], fill=255)
    drawSurface.line(landmark72, fill=255,width=5)
    image.save(img_path.replace('.jpg', 'res.png'))
    image.show()


def predict(trX):
    # file = '../data/face_top_9.mat'
    # data = scio.loadmat(file)
    tf.reset_default_graph()
    # graph
    saver = tf.train.import_meta_graph("save/model-1000-2.ckpt.meta")
    # value
    # a = tf.train.NewCheckpointReader('save/model.ckpt.index')
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "save/model-1000-2.ckpt")
        graph = tf.get_default_graph()

        predict_op = graph.get_tensor_by_name("output/BiasAdd:0")
        X = graph.get_tensor_by_name("Placeholder:0")
        # dp = graph.get_tensor_by_name("Placeholder_2:0")

        resY = predict_op.eval({X: trX.reshape(1, -1) / 255.})
        # resY=[[31,10]]
    print(resY)
    # resY = [[14.34780979, 32.37727928, 17.39715767, 22.06736565, 23.70981216,
    #          17.21895123, 29.31753731, 16.67663288, 31.93413925, 14.36086273,
    #          48.92932129, 29.01085472, 45.96300888, 21.74747467, 42.84361649,
    #          17.86888313, 34.78334045, 14.6940918]]
    return resY


if __name__ == '__main__':
    main()
