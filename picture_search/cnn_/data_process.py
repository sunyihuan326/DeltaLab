# coding:utf-8 
'''
created on 2018/10/16

@author:sunyihuan
'''
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
import numpy as np

batch_size = 32
img_height = 224
img_depth = 3


def loss_with_spring(o1, o2):
    eucd2 = tf.pow(tf.subtract(o1, o2), 2)
    eucd2 = tf.reduce_sum(eucd2, 1)
    eucd = tf.sqrt(eucd2 + 1e-6, name="eucd")
    loss = eucd
    return loss


def parser(record, aug):
    keys_to_features = {
        'image_path': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    # image_raw = tf.read_file(FLAGS.data_dir_complex + '/' + parsed['image_path'])  # 图片名为image_path
    image_raw = tf.read_file(parsed['image_path'])  # image_path为全路径
    image = tf.image.decode_jpeg(image_raw)
    image = tf.image.resize_images(image, [img_height, img_height])
    image = tf.reshape(image, [img_height, img_height, img_depth])
    image = tf.cast(image, tf.float32)

    if aug:
        # image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=1, upper=1.2)  # 随机调整图片对比度
        # image = tf.image.random_brightness(image, max_delta=0.2)  # 随机调整图片亮度
        image = tf.image.random_flip_left_right(image)  # 随机左右翻转
        # image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)  # 图片标准化
    label = tf.cast(parsed['label'], tf.int32)
    return image, label, parsed['image_path']


def get_record_data(graph, data_dir, typ, aug=False):
    with graph.as_default():
        dataset = tf.data.TFRecordDataset(os.path.join(data_dir, '{}.tfrecords'.format(typ)))

        dataset = dataset.map(lambda x: parser(x, aug))

        # num_epochs = -1 if typ == 'train' else 1
        num_epochs = 1
        # buffer_size = _NUM_IMAGES[typ]
        dataset = dataset.repeat(num_epochs).shuffle(buffer_size=batch_size).batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

        features, labels, paths = iterator.get_next()
        # features = (features - 128) / 68.0
        return features, labels, paths


def valid_result_analysis(tp, ckpt_path, data_path):
    '''
    :param tp: syh电脑还是台式
    :param ckpt_path: 模型参数文件
    :param data_path: 数据地址
    :param error_show: 是否保存预测错误的图片
    :return:
    '''
    if tp == "syh":
        error_path = '/Users/sunyihuan/PycharmProjects/delta/hairstyle/hair_length2/error'
    else:
        error_path = 'D:\project\delta\hairstyle\hair_length2\error'
    if not os.path.exists(error_path):
        os.makedirs(error_path)

    graph = tf.get_default_graph()
    saver = tf.train.import_meta_graph("{}.meta".format(ckpt_path))
    sess = tf.InteractiveSession(graph=graph)
    saver.restore(sess, ckpt_path)
    # predict_op = graph.get_tensor_by_name("classes:0")
    out = graph.get_tensor_by_name("DENSE2: 0")
    features = graph.get_tensor_by_name("features: 0")

    te_features, te_labels, te_paths = get_record_data(graph, data_path, 'test')

    for _ in range(1):
        try:
            te_feature_batch2, te_label_batch2, te_path_batch2 = sess.run([te_features, te_labels, te_paths])
            te_feature_batch2 = te_feature_batch2.reshape(-1, img_height, img_height, 3)
            print(te_label_batch2)

            out__ = out.eval(feed_dict={features: te_feature_batch2})
            print(out__.shape)

        except tf.errors.OutOfRangeError:
            print(_)


if __name__ == "__main__":
    ckpt_path = "/Users/sunyihuan/Desktop/parameters/hair_length2/test77.9/model.ckpt-14"

    bei_data_path = "D:/material/hair_len/0908zheng"

    valid_result_analysis("sy", ckpt_path, bei_data_path)