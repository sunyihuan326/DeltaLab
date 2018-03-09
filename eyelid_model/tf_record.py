# coding:utf-8
'''
Created on 2018/1/31.

@author: chk01
'''

import os
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import pickle
from tqdm import tqdm


class Data2TFrecord(object):
    def __init__(self, source_dir):
        self.source_dir = source_dir
        self.__load_xy()
        self.image = ''

    def __load_xy(self):
        self.images_path = []
        self.labels = []
        self.classes = []
        folders = os.listdir(self.source_dir)
        for folder in folders:
            self.classes.append(folder)
            path_folder = os.path.join(self.source_dir, folder)
            for root, sub_folder, files in os.walk(path_folder):
                for file in files:
                    image_path = self.source_dir + "/" + folder + "/" + file
                    self.images_path.append(image_path)
                    self.labels.append(folder)

    def save(self, save_path):
        to_gray = True
        rotate = False
        brightness = False
        contrast = False
        size = 64
        epoches = 1

        number_image = len(self.images_path)
        file_queue = tf.train.string_input_producer(self.images_path, shuffle=False,num_epochs=1)
        image_reader = tf.WholeFileReader()
        key, image = image_reader.read(file_queue)
        self.image = tf.image.decode_jpeg(image)

        # if rgb_to_grayscale
        if to_gray:
            self.image = (tf.image.rgb_to_grayscale(self.image))

        # rotate_image
        if rotate:
            angle = 1 * (np.random.random() - 0.5) / 5
            self.image = tf.contrib.image.rotate(self.image, angle, interpolation='BILINEAR')

        # resize_images
        self.image = tf.image.resize_images(self.image, [size, size], method=0)

        # random_brightness
        if brightness:
            self.image = tf.image.random_brightness(self.image, max_delta=0.3)
        # random_contrast
        if contrast:
            self.image = tf.image.random_contrast(self.image, lower=20, upper=99)
        # crop_to_bounding_box 暂时用dropout代替

        self.image = tf.cast(tf.squeeze(self.image), tf.float32) / 255.0

        with tf.Session() as sess:
            label_dict = dict(zip(self.classes, range(len(self.classes))))
            # with open('class2label', 'wb') as f:
            #     pickle.dump(label_dict, f)
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            writer = tf.python_io.TFRecordWriter(save_path)

            coord = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in tqdm(range(number_image), ncols=70, leave=False, unit='b'):
                label = label_dict[self.labels[i]]
                for j in range(epoches):
                    image_i = sess.run(self.image)
                    # image_i = image_i - image_means
                    # todo 去中心化
                    image_raw = image_i.tobytes()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'label': self.__int64_feature(label),
                        'image_raw': self.__bytes_feature(image_raw)}))
                    writer.write(example.SerializeToString())
            writer.close()
            coord.request_stop()
            coord.join(threads)

    def __int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def __bytes_feature(self, value):
        # tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read(path):
    batch_size = 64
    image_size = 64
    filename_queue = tf.train.string_input_producer([path], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(img_features['image_raw'], tf.float32)
    # image = tf.cast(image, tf.float32)
    min_after_dequeue = 1000
    image = tf.reshape(image, [image_size, image_size, 1])
    # image = tf.cast(image, tf.float32)
    label = tf.cast(img_features['label'], tf.int32)
    capacity = min_after_dequeue + 3 * batch_size
    print(4)
    image_batch, label_batch = tf.train.batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=1,
                                                      capacity=capacity,
                                                      )
    print(5)
    return image_batch, label_batch


# image = tf.image.per_image_standardization(image)
# image_batches,label_batches = tf.train.batch([image, label], batch_size=16, capacity=20)
# img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=30, capacity=2000, min_after_dequeue=1000)
# 队列中允许的数据最大量capacity>=min_after_dequeue+num_threads*batch_size。
# F:/dataSets/CASIA/HWDB1/train/train

tt = Data2TFrecord('C:/Users/chk01/Desktop/eyelid/fix1/train')
tt.save('model/data/train1.tfrecords')
assert 1 == 0
# C:/Users/chk01/Desktop/eyelid/data/
image_batch_j, label_batch_j = read('model/data/valid1.tfrecords')
# print(6)
with tf.Session() as sess:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for j in range(10):
            image_batch_now, label_batch_now = sess.run([image_batch_j, label_batch_j])
            for i in range(len(image_batch_now)):
                if i == 0:
                    plt.figure()
                    plt.axis('equal')
                    plt.imshow(image_batch_now[i].reshape(64, 64), cmap='gray')
                    plt.title(label_batch_now[i])
                    plt.show()
    except tf.errors.OutOfRangeError:
        print("Done reading!")
        coord.request_stop()
    finally:
        coord.request_stop()
        coord.join(threads)
