# coding:utf-8 
'''
created on 2018/7/14

@author:sunyihuan
'''

from SE_Net.seNet import *

import sys
import tensorflow as tf
from tensorflow.contrib.slim.nets import alexnet, overfeat, inception, vgg, resnet_v2
import tensorflow.contrib.slim as slim
import os
from tqdm import tqdm
import numpy as np

flags = tf.flags
flags.DEFINE_bool('train', True, 'Either train the model or test the model.')
# flags.DEFINE_integer('num_epochs', 100, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 2, 'Batch size')

flags.DEFINE_integer('train_steps', 200, 'steps of training')
flags.DEFINE_integer('test_steps', 200, 'steps of testing')
flags.DEFINE_string('test_data', "test", 'typ of testing')
flags.DEFINE_integer('checkpoint_epoch', 1000, 'The checkpoint epoch')

flags.DEFINE_integer('img_height', 224, 'height size')
flags.DEFINE_integer('img_depth', 3, 'depth size')
flags.DEFINE_integer('num_classes', 9, 'num of classes')

flags.DEFINE_float('min_lr', 0.01, 'the minimum value of  lr')
flags.DEFINE_float('start_lr', 0.4, 'the start value of  lr')
flags.DEFINE_float('dropout_rate', .5, 'Dropout rate')
flags.DEFINE_float('weight_decay', .005, 'weight_decay of l2_loss')
flags.DEFINE_float('momentum', .9, 'the momentum ')

flags.DEFINE_string('data_dir',
                    '/Users/sunyihuan/Desktop/Data/hair_length/complex_and_simple_train0712/tfrecord/on_enhance_only_path',
                    'The data directory.')

flags.DEFINE_string('data_dir_simple',
                    '/Users/sunyihuan/Desktop/Data/hair_length/hair_last/model_using_data0/tfrecord/0711_no_enhance_path224',
                    'The data directory.')
flags.DEFINE_string('data_dir_complex',
                    '/Users/sunyihuan/Desktop/Data/hair_length/complex_pic/hair_length_pasted_tfrecord/224_0709_onlyPath',
                    'The data directory.')

flags.DEFINE_string('summary_dir',
                    '/Users/sunyihuan/PycharmProjects/DeltaLab/SE_Net/seNet_ckpt',
                    'The summary dir')
FLAGS = flags.FLAGS

_NUM_IMAGES = {
    'train': 100,
    'valid0': 500,
    'valid1': 500,
    'valid': 500,
}


def parser(record, aug):
    keys_to_features = {
        'image_path': tf.FixedLenFeature((), tf.string),
        'label': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    # image_raw = tf.read_file(FLAGS.data_dir_complex + '/' + parsed['image_path'])  # 图片名为image_path
    image_raw = tf.read_file(parsed['image_path'])  # image_path为全路径
    image = tf.image.decode_jpeg(image_raw)
    image = tf.image.resize_images(image, [FLAGS.img_height, FLAGS.img_height])
    image = tf.reshape(image, [FLAGS.img_height, FLAGS.img_height, FLAGS.img_depth])
    image = tf.cast(image, tf.float32)
    # image = tf.decode_raw(parsed['image_raw'], tf.uint8)
    # image = tf.reshape(image, [FLAGS.img_height, FLAGS.img_height, FLAGS.img_depth])

    # image = tf.random_crop(image, [124, 124, 3])
    if aug:
        # image = tf.image.random_hue(image, max_delta=0.05)
        # image = tf.image.random_contrast(image, lower=1, upper=1.2)  # 随机调整图片对比度
        image = tf.image.random_brightness(image, max_delta=0.2)  # 随机调整图片亮度
        # image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)  # 图片标准化
    label = tf.cast(parsed['label'], tf.int32)
    return image, label, parsed['image_path']


def get_record_data(graph, data_dir, typ, aug=False):
    with graph.as_default():
        if data_dir == 'all':
            dataset = tf.data.TFRecordDataset([os.path.join(FLAGS.data_dir_complex, '{}.tfrecords'.format(typ)),
                                               os.path.join(FLAGS.data_dir_simple, '{}.tfrecords'.format(typ)),
                                               os.path.join(FLAGS.data_dir_complex, '{}.tfrecords'.format(typ))])
        else:
            dataset = tf.data.TFRecordDataset(os.path.join(data_dir, '{}.tfrecords'.format(typ)))

        dataset = dataset.map(lambda x: parser(x, aug))

        num_epochs = -1 if typ == 'train' else 1
        batch_size = FLAGS.batch_size
        # buffer_size = _NUM_IMAGES[typ]
        dataset = dataset.repeat(num_epochs).shuffle(buffer_size=batch_size).batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

        features, labels, paths = iterator.get_next()
        # features = (features - 128) / 68.0
        return features, labels, paths


def extract_step(path):
    file_name = os.path.basename(path)
    return int(file_name.split('-')[-1])


def loader(saver, session, load_dir):
    if tf.gfile.Exists(load_dir):
        ckpt = tf.train.get_checkpoint_state(load_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            prev_step = extract_step(ckpt.model_checkpoint_path)

        else:
            tf.gfile.DeleteRecursively(load_dir)
            tf.gfile.MakeDirs(load_dir)
            prev_step = 0
    else:
        tf.gfile.MakeDirs(load_dir)
        prev_step = 0
    return prev_step


def finetune_model(base_out, num_classes, drop_rate):
    activation = tf.nn.relu

    reg1 = tf.contrib.layers.l2_regularizer(scale=1.6)
    net = tf.layers.conv2d(base_out, 32, 3, padding="same", activation=activation, kernel_regularizer=reg1)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='valid')
    net = tf.layers.batch_normalization(net)
    #
    net = tf.layers.conv2d(net, 64, 3, padding='valid', activation=activation, kernel_regularizer=reg1)
    net = tf.layers.conv2d(net, 64, 3, padding='same', activation=activation, kernel_regularizer=reg1)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='valid')
    net = tf.layers.batch_normalization(net)

    net = tf.layers.conv2d(net, 128, 3, padding='same', activation=activation, kernel_regularizer=reg1)
    net = tf.layers.conv2d(net, 128, 3, padding='same', activation=activation, kernel_regularizer=reg1)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='valid')
    net = tf.layers.batch_normalization(net)

    net = tf.layers.conv2d(net, 256, 3, padding='valid', activation=activation, kernel_regularizer=reg1)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='valid')

    net = tf.layers.conv2d(net, 512, 3, padding='valid', activation=activation, kernel_regularizer=reg1)
    net = tf.layers.conv2d(net, 512, 3, padding='same', activation=activation)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='valid')

    net = tf.layers.conv2d(net, 256, 3, padding='same', activation=activation, kernel_regularizer=reg1)
    net = tf.layers.conv2d(net, 256, 3, padding='same', activation=activation)
    net = tf.layers.max_pooling2d(net, 2, 2, padding='valid')

    net = tf.layers.flatten(net)
    net = tf.layers.dropout(net, drop_rate)
    net = tf.layers.dense(inputs=net, units=500, name='DENSE1')
    net = tf.layers.dropout(net, drop_rate)
    net = tf.layers.dense(inputs=net, units=100, name='DENSE2')
    out = tf.layers.dense(inputs=net, units=num_classes, name='OUT')
    return out


def summary(var_list, loss, out, Y, learning_rate, load_dir):
    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('cross_entropy', loss)

    # Evaluation op: Accuracy of the model
    with tf.name_scope("accuracy"):
        predict_op = tf.argmax(out, 1)
        print(predict_op)
        correct_pred = tf.equal(predict_op, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Add the learning_rate to summary
    tf.summary.scalar('learning_rate', learning_rate)

    # Add the accuracy to the summary
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries together
    merge_op = tf.summary.merge_all()

    return merge_op, accuracy


def param_report(graph):
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        graph, tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)


class Model(object):
    def __init__(self, inputs, num_classes=9):
        self.logits = inputs


def train():
    graph = tf.Graph()
    with graph.as_default():
        class_num = FLAGS.num_classes
        drop_rate = FLAGS.dropout_rate
        lr = FLAGS.start_lr
        weight_decay = FLAGS.weight_decay
        momentum = FLAGS.momentum
        load_dir = FLAGS.summary_dir + '/train/'

        # features, labels = get_record_data(graph, 'train')
        features = tf.placeholder(tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_height, FLAGS.img_depth],
                                  name='features')
        print(features)
        labels = tf.placeholder(tf.int32, shape=[None, ], name='labels')
        Y = tf.one_hot(labels, depth=class_num, axis=1, dtype=tf.float32)

        training_flag = tf.placeholder(tf.bool)

        logits = SE_Inception_v4(features, training=training_flag).model
        out = logits
        predict_op = tf.argmax(input=out, axis=1, name='classes')

        # param_report(graph)  # 参数的总数

        cost = tf.losses.softmax_cross_entropy(Y, logits, scope='cost', weights=1.0)
        var_list = []

        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        loss = cost + l2_loss * weight_decay

        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        learning_rate = tf.train.exponential_decay(lr, global_step=global_step, decay_steps=1000, decay_rate=0.9)
        learning_rate = tf.maximum(learning_rate, FLAGS.min_lr)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
        train_op = optimizer.minimize(cost + l2_loss * weight_decay)

        # param_report
        # param_report(graph)

        merge_op, accuracy = summary(var_list, loss, out, Y, learning_rate, load_dir)
        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver(max_to_keep=20)

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = int(np.floor(_NUM_IMAGES['train'] / FLAGS.batch_size))
        val_batches_per_epoch = int(np.floor(_NUM_IMAGES['valid'] / FLAGS.batch_size))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            # tr_features, tr_labels, _ = get_record_data(graph, 'train', True)
            # Loop over number of epochs
            last_step = loader(saver, sess, load_dir)
            max_steps = FLAGS.train_steps

            # Initialize the FileWriter
            train_writer = tf.summary.FileWriter(load_dir, graph=sess.graph)
            valid_writer0 = tf.summary.FileWriter(load_dir.replace("train", "valid0"), graph=sess.graph)
            valid_writer1 = tf.summary.FileWriter(load_dir.replace("train", "valid1"), graph=sess.graph)
            valid_writer = tf.summary.FileWriter(load_dir.replace("train", "valid"), graph=sess.graph)

            for epoch in range(last_step, max_steps):
                # tr_features, tr_labels = get_record_data(graph, 'train')
                # te_features, te_labels = get_record_data(graph, 'valid')
                print('epoch:', epoch + 1, '/', "{}".format(max_steps))
                train_acc = 0
                train_count = 0
                tr_features, tr_labels, _ = get_record_data(graph, "all", "train", True)
                # if epoch % 2 == 0:
                #     tr_features, tr_labels, _ = get_record_data(graph, FLAGS.data_dir_simple, 'train', True)
                # else:
                #     tr_features, tr_labels, _ = get_record_data(graph, FLAGS.data_dir_complex, 'train', True)

                for step in tqdm(range(train_batches_per_epoch)):
                    # get next batch of data
                    tr_feature_batch, tr_label_batch = sess.run([tr_features, tr_labels])

                    train_feed_dict = {
                        features: tr_feature_batch,
                        labels: tr_label_batch,
                        training_flag: True
                    }

                    _, acc = sess.run([train_op, accuracy], feed_dict=train_feed_dict)
                    # Generate summary with the current batch of data and write to file
                    if step % 2 == 0:
                        s = sess.run(merge_op, feed_dict=train_feed_dict)
                        train_writer.add_summary(s, epoch * train_batches_per_epoch + step)

                    train_acc += acc * len(tr_label_batch)
                    train_count += len(tr_label_batch)

                    del tr_feature_batch
                    del tr_label_batch

                train_acc /= train_count
                _S0 = tf.Summary(value=[tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
                train_writer.add_summary(summary=_S0, global_step=epoch)
                print("Training Accuracy = {:.4f}".format(train_acc))

                test_acc_simple0 = 0
                test_count_simple0 = 0
                test_acc_simple1 = 0
                test_count_simple1 = 0
                test_acc_complex = 0
                test_count_complex = 0
                te_features_simple_valid0, te_labels_simple_valid0, _0 = get_record_data(graph, FLAGS.data_dir_simple,
                                                                                         'valid0', False)
                te_features_simple_valid1, te_labels_simple_valid1, _1 = get_record_data(graph, FLAGS.data_dir_simple,
                                                                                         'valid1', False)
                te_features_complex, te_labels_complex, _ = get_record_data(graph, FLAGS.data_dir_complex, 'valid',
                                                                            False)
                for _ in range(31):

                    try:
                        te_feature_batch0, te_label_batch0 = sess.run(
                            [te_features_simple_valid0, te_labels_simple_valid0])
                        test_feed_dict0 = {
                            features: te_feature_batch0,
                            labels: te_label_batch0,
                            training_flag: False
                        }
                        acc = sess.run(accuracy, feed_dict=test_feed_dict0)
                        s = sess.run(merge_op, feed_dict=test_feed_dict0)
                        valid_writer0.add_summary(s, epoch * 130 + _)

                        test_acc_simple0 += acc * len(te_label_batch0)
                        test_count_simple0 += len(te_label_batch0)
                        del te_feature_batch0
                        del te_label_batch0
                    except tf.errors.OutOfRangeError:
                        print(_)
                test_acc_simple0 /= test_count_simple0
                _S1 = tf.Summary(value=[tf.Summary.Value(tag='simple_accuracy0', simple_value=test_acc_simple0)])
                valid_writer0.add_summary(summary=_S1, global_step=epoch)
                print("Validation0 Accuracy simple= {:.4f}".format(test_acc_simple0))

                for _ in range(31):
                    try:
                        te_feature_batch1, te_label_batch1 = sess.run(
                            [te_features_simple_valid1, te_labels_simple_valid1])

                        test_feed_dict1 = {
                            features: te_feature_batch1,
                            labels: te_label_batch1,
                            training_flag: False
                        }

                        acc = sess.run(accuracy, feed_dict=test_feed_dict1)
                        s = sess.run(merge_op, feed_dict=test_feed_dict1)
                        valid_writer1.add_summary(s, epoch * 130 + _)

                        test_acc_simple1 += acc * len(te_label_batch1)
                        test_count_simple1 += len(te_label_batch1)
                        del te_feature_batch1
                        del te_label_batch1
                    except tf.errors.OutOfRangeError:
                        print(_)
                test_acc_simple1 /= test_count_simple1
                _S1 = tf.Summary(value=[tf.Summary.Value(tag='simple_accuracy1', simple_value=test_acc_simple1)])
                valid_writer1.add_summary(summary=_S1, global_step=epoch)
                print("Validation1 Accuracy simple= {:.4f}".format(test_acc_simple1))

                for _ in range(16):
                    try:
                        te_feature_batch, te_label_batch = sess.run([te_features_complex, te_labels_complex])
                        test_feed_dict = {
                            features: te_feature_batch,
                            labels: te_label_batch,
                            training_flag: False
                        }
                        acc = sess.run(accuracy, feed_dict=test_feed_dict)
                        s = sess.run(merge_op, feed_dict=test_feed_dict)
                        valid_writer.add_summary(s, epoch * 130 + _)

                        test_acc_complex += acc * len(te_label_batch)
                        test_count_complex += len(te_label_batch)
                        del te_feature_batch
                        del te_label_batch
                    except tf.errors.OutOfRangeError:
                        print(_)
                test_acc_complex /= test_count_complex
                _S2 = tf.Summary(value=[tf.Summary.Value(tag='complex_accuracy', simple_value=test_acc_complex)])
                valid_writer.add_summary(summary=_S2, global_step=epoch)
                print("Validation Accuracy complex = {:.4f}".format(test_acc_complex))

                # if (epoch + 1) % 2 == 0:
                # save checkpoint of the model
                saver.save(sess, os.path.join(load_dir, 'model.ckpt'), global_step=epoch + 1)

            train_writer.close()
            valid_writer0.close()
            valid_writer.close()


def main(_):
    if FLAGS.train:
        train()
    else:
        pass
        # predict()


if __name__ == '__main__':
    tf.app.run()
