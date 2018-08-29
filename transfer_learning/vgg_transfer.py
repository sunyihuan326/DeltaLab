# coding:utf-8 
'''
created on 2018/8/20

@author:sunyihuan
'''

import sys
import tensorflow as tf
from tensorflow.contrib.slim.nets import alexnet, overfeat, inception, vgg, resnet_v2, resnet_v1
import tensorflow.contrib.slim as slim
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image
from functools import reduce

from nets import nets_factory

flags = tf.flags
flags.DEFINE_bool('train', True, 'Either train the model or test the model.')
# flags.DEFINE_integer('num_epochs', 100, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 32, 'Batch size')

flags.DEFINE_integer('train_steps', 200, 'steps of training')
flags.DEFINE_integer('test_steps', 200, 'steps of testing')
flags.DEFINE_string('test_data', "test", 'typ of testing')
flags.DEFINE_integer('checkpoint_epoch', 1000, 'The checkpoint epoch')

flags.DEFINE_integer('img_height', 224, 'height size')
flags.DEFINE_integer('img_depth', 3, 'depth size')
flags.DEFINE_integer('num_classes', 9, 'num of classes')

flags.DEFINE_float('min_lr', 1e-6, 'the minimum value of  lr')
flags.DEFINE_float('start_lr', 1e-4, 'the start value of  lr')
flags.DEFINE_float('weight_decay', 1e-8, 'weight_decay of l2_loss')
flags.DEFINE_float('dropout_rate', .5, 'Dropout rate')

flags.DEFINE_string('data_dir_simple',
                    '/Users/sunyihuan/Desktop/Data/hair_length/second_data_for_model/simple',
                    'The data directory.')
flags.DEFINE_string('data_dir_complex',
                    '/Users/sunyihuan/Desktop/Data/hair_length/second_data_for_model/complex_all',
                    'The data directory.')
flags.DEFINE_string('data_dir_all_train',
                    '/Users/sunyihuan/Desktop/Data/hair_length/second_data_for_model/all_train',
                    'The data directory.')

flags.DEFINE_string('summary_dir',
                    '/Users/sunyihuan/PycharmProjects/myself/DeltaLab/transfer_learning/ckpt',
                    'The summary dir')
flags.DEFINE_string('checkpoint_dir', 'C:/Users/admin/Desktop/data/face3/tmp/74.12', 'The checkpoint dir')
FLAGS = flags.FLAGS

_NUM_IMAGES = {
    'train': 100,
    'valid0': 100,
    'valid1': 100,
    'valid2': 100,
    "valid3": 140,
    "test": 100
}

trained_model_dir = '/Users/sunyihuan/Desktop/modelParameters/model'


class Model(object):
    def __init__(self, net, inputs):
        # inputs = tf.random_uniform((21, 224, 224, 3))
        model_typ = net.split('_')[0]
        pre_trained_model = os.path.join(trained_model_dir, model_typ, net + '.ckpt')
        net_fn = nets_factory.get_network_fn(net)

        self.logits, self.end_points = net_fn(inputs)

        exclude = nets_factory.excludes_map[net]
        variables_to_restore = slim.get_variables_to_restore(
            # include=["vgg_16/conv1/conv1_1", "vgg_16/conv1/conv1_2", "vgg_16/pool1", "vgg_16/conv2/conv2_1"],
            exclude=exclude)

        # for v in variables_to_restore:
        #     print(v.name.split(":")[0])
        tf.train.init_from_checkpoint(pre_trained_model + '',
                                      {v.name.split(':')[0]: v for v in variables_to_restore})
        self.saver = tf.train.Saver(max_to_keep=300)


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
        image = tf.image.random_contrast(image, lower=1, upper=1.2)  # 随机调整图片对比度
        image = tf.image.random_brightness(image, max_delta=0.2)  # 随机调整图片亮度
        # image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)  # 图片标准化
    label = tf.cast(parsed['label'], tf.int32)
    return image, label, parsed['image_path']


def get_record_data(graph, data_dir, typ, aug=False):
    with graph.as_default():
        if data_dir == 'all':
            dataset = tf.data.TFRecordDataset([os.path.join(FLAGS.data_dir_all_train, '{}.tfrecords'.format(typ))])
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
    print(file_name)
    return int(file_name.split('-')[-1])


def loader(saver, session, load_dir):
    print('load_dir', load_dir)
    print(tf.gfile.Exists(load_dir))
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
    net = tf.layers.flatten(base_out)
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


def train():
    graph = tf.Graph()
    with graph.as_default():
        class_num = FLAGS.num_classes
        drop_rate = FLAGS.dropout_rate
        lr = FLAGS.start_lr
        load_dir = FLAGS.summary_dir + '/train/'

        # features, labels = get_record_data(graph, 'train')
        features = tf.placeholder(tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_height, FLAGS.img_depth],
                                  name='features')
        print(features)
        labels = tf.placeholder(tf.int32, shape=[None, ], name='labels')
        Y = tf.one_hot(labels, depth=class_num, axis=1, dtype=tf.float32)

        # extend others model
        model = Model('vgg_16', features)
        # for key in model.end_points.keys():
        #     print(key)
        # 根据模型需要修改
        net = model.end_points["vgg_16/conv2/conv2_1"]

        # 后续的模型结构
        out = finetune_model(net, class_num, drop_rate)  # OUT/BiasAdd:0
        # print("out:", out)  # OUT/BiasAdd:0    shape=(?, 9)
        predict_op = tf.argmax(input=out, axis=1, name='classes')

        # tf.trainable_variables()计算总参数量

        size = lambda v: reduce(lambda x, y: x * y, v.get_shape().as_list())
        n = sum(size(v) for v in tf.trainable_variables())
        print("trainable_variables:", n)

        # param_report(graph)  # 参数的总数

        loss = tf.losses.softmax_cross_entropy(Y, out, scope='LOSS', weights=1.0)

        # for v in tf.trainable_variables():
        #     print(v)
        train_layers = ["vgg_16/conv1", "vgg_16/conv2", "OUT"]
        var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
        # print(var_list)
        print(var_list)

        global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
        print("global_step:", global_step)
        learning_rate = tf.train.exponential_decay(lr, global_step=global_step, decay_steps=100, decay_rate=0.9)
        learning_rate = tf.maximum(learning_rate, FLAGS.min_lr)

        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

        # param_report
        # param_report(graph)

        merge_op, accuracy = summary(var_list, loss, out, Y, learning_rate, load_dir)
        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver(var_list=var_list, max_to_keep=20)

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = int(np.floor(_NUM_IMAGES['train'] / FLAGS.batch_size))
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
            valid_writer = tf.summary.FileWriter(load_dir.replace("train", "valid"), graph=sess.graph)

            for epoch in range(last_step, max_steps):
                # tr_features, tr_labels = get_record_data(graph, 'train')
                # te_features, te_labels = get_record_data(graph, 'valid')
                print('epoch:', epoch + 1, '/', "{}".format(max_steps))
                train_acc = 0
                train_count = 0
                loss_tr = 0.
                tr_features, tr_labels, _ = get_record_data(graph, "all", "train", True)
                for step in tqdm(range(train_batches_per_epoch)):
                    # get next batch of data
                    tr_feature_batch, tr_label_batch = sess.run([tr_features, tr_labels])

                    train_feed_dict = {
                        features: tr_feature_batch,
                        labels: tr_label_batch
                    }

                    _, acc, loss_, = sess.run([train_op, accuracy, loss], feed_dict=train_feed_dict)
                    # Generate summary with the current batch of data and write to file
                    if step % 2 == 0:
                        s = sess.run(merge_op, feed_dict=train_feed_dict)
                        train_writer.add_summary(s, epoch * train_batches_per_epoch + step)

                    train_acc += acc * len(tr_label_batch)
                    train_count += len(tr_label_batch)
                    loss_tr += loss_ * len(tr_label_batch)

                    del tr_feature_batch
                    del tr_label_batch

                train_acc /= train_count
                loss_tr /= train_count
                _S0 = tf.Summary(value=[tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
                train_writer.add_summary(summary=_S0, global_step=epoch)
                print("Training Accuracy = {:.7f}".format(train_acc))
                print("Training loss = {:.7f}".format(loss_tr))

                test_acc_complex = 0
                test_count_complex = 0
                te_features_complex, te_labels_complex, _ = get_record_data(graph, FLAGS.data_dir_complex, "test",
                                                                            False)

                for _ in range(24):
                    try:
                        te_feature_batch, te_label_batch = sess.run([te_features_complex, te_labels_complex])
                        test_feed_dict = {
                            features: te_feature_batch,
                            labels: te_label_batch
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
                _S1 = tf.Summary(value=[tf.Summary.Value(tag='test_complex_accuracy', simple_value=test_acc_complex)])
                valid_writer.add_summary(summary=_S1, global_step=epoch)
                print("test complex Accuracy = {:.4f}".format(test_acc_complex))

                # # if (epoch + 1) % 2 == 0:
                # # save checkpoint of the model
                saver.save(sess, os.path.join(load_dir, 'model.ckpt'), global_step=epoch + 1)

            train_writer.close()
            valid_writer.close()


def predict(error_show=False):
    pass


def main(_):
    if FLAGS.train:
        train()
    else:
        predict(error_show=True)


if __name__ == '__main__':
    tf.app.run()
