# coding:utf-8 
'''
created on 2018/8/22

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

from tensorflow.python import pywrap_tensorflow

from nets import nets_factory

flags = tf.flags
flags.DEFINE_bool('train', True, 'Either train the model or test the model.')
# flags.DEFINE_integer('num_epochs', 100, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 32, 'Batch size')

flags.DEFINE_integer('model_version', 1, 'model version')
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
                    '/Users/sunyihuan/PycharmProjects/myself/DeltaLab/serving_learning/model_save',
                    'The summary dir')
flags.DEFINE_string('checkpoint_dir', 'C:/Users/admin/Desktop/data/face3/tmp/74.12', 'The checkpoint dir')
FLAGS = flags.FLAGS

_NUM_IMAGES = {
    'train': 120,
    'valid0': 20,
    'valid1': 20,
    'valid2': 20,
    "valid3": 20,
    "test": 20
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
        feature_configs = {'features': tf.FixedLenFeature(shape=[224, 224, 3], dtype=tf.float32), }
        # features = tf.placeholder(tf.float32, shape=[None, FLAGS.img_height, FLAGS.img_height, FLAGS.img_depth],
        #                           name='features')

        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        features = tf_example['features']
        labels = tf.placeholder(tf.int32, shape=[None, ], name='labels')
        Y = tf.one_hot(labels, depth=class_num, axis=1, dtype=tf.float32)

        # extend others model
        model = Model('inception_v4', features)
        for key in model.end_points.keys():
            print(key)
        # 根据模型需要修改
        net = model.end_points["global_pool"]

        # 后续的模型结构
        out = finetune_model(net, class_num, drop_rate)  # OUT/BiasAdd:0
        # print("out:", out)  # OUT/BiasAdd:0    shape=(?, 9)
        predict_op = tf.argmax(input=out, axis=1, name='classes')

        values, indices = tf.nn.top_k(predict_op, FLAGS.num_classes)
        table = tf.contrib.lookup.index_to_string_table_from_tensor(
            tf.constant([str(i) for i in range(FLAGS.num_classes)]))
        prediction_classes = table.lookup(tf.to_int64(indices))

        # tf.trainable_variables()计算总参数量

        size = lambda v: reduce(lambda x, y: x * y, v.get_shape().as_list())
        n = sum(size(v) for v in tf.trainable_variables())
        print("trainable_variables:", n)

        # param_report(graph)  # 参数的总数

        loss = tf.losses.softmax_cross_entropy(Y, out, scope='LOSS', weights=1.0)

        for v in tf.global_variables():
            print(v)

        # Initialize an saver for store model checkpoints
        # saver = tf.train.Saver(var_list=var_list, max_to_keep=20)

        # Get the number of training/validation steps per epoch
        train_batches_per_epoch = int(np.floor(_NUM_IMAGES['train'] / FLAGS.batch_size))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            # sess.run(tf.global_variables_initializer())
            # restore model from checkpoints
            model_path = "/Users/sunyihuan/Desktop/parameters/hair_length/second/complex77_simple70/model.ckpt-39"
            saver = tf.train.import_meta_graph("{}.meta".format(model_path))
            saver.restore(sess, model_path)

            # saver = tf.train.Saver()
            # module_file = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            # saver.restore(sess, module_file)

            builder = tf.saved_model.builder.SavedModelBuilder(
                os.path.join(load_dir, str(FLAGS.model_version)))

            tensor_info_x = tf.saved_model.utils.build_tensor_info(features)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(predict_op)
            signature_def_map = {
                "predict_images": tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={"images": tensor_info_x},
                    outputs={"scores": tensor_info_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            }
            builder.add_meta_graph_and_variables(sess,
                                                 [tf.saved_model.tag_constants.SERVING],
                                                 signature_def_map=signature_def_map)
            builder.save()

            # Export model
            # WARNING(break-tutorial-inline-code): The following code snippet is
            # in-lined in tutorials, please update tutorial documents accordingly
            # whenever code changes.
            # export_path_base = load_dir
            # export_path = os.path.join(
            #     tf.compat.as_bytes(export_path_base),
            #     tf.compat.as_bytes(str(FLAGS.model_version)))
            # print('Exporting trained model to', export_path)
            # builder = tf.saved_model.builder.SavedModelBuilder(export_path)
            #
            # # create tensors info
            # predict_tensor_inputs_info = tf.saved_model.utils.build_tensor_info(features)
            # predict_tensor_scores_info = tf.saved_model.utils.build_tensor_info(
            #         predict_op)
            #
            # # build prediction signature
            # prediction_signature = (
            #         tf.saved_model.signature_def_utils.build_signature_def(
            #             inputs={'images': predict_tensor_inputs_info},
            #             outputs={'scores': predict_tensor_scores_info},
            #             method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            #         )
            #     )
            #
            #     # save the model
            # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            # builder.add_meta_graph_and_variables(
            #         sess, [tf.saved_model.tag_constants.SERVING],
            #         signature_def_map={
            #             'predict_images': prediction_signature
            #         },
            #         legacy_init_op=legacy_init_op)
            #
            # builder.save()


def predict(error_show=False):
    pass


def main(_):
    if FLAGS.train:
        train()
    else:
        predict(error_show=True)


if __name__ == '__main__':
    tf.app.run()
