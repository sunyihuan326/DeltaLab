# coding:utf-8 
'''
created on 2018/9/1

@author:sunyihuan
'''
import tensorflow as tf
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score
# from hair_length.script.data_distribution import get_record_data
from transfer_learning.vgg_transfer import get_record_data
import numpy as np
from PIL import Image
import os


def valid_result_analysis(ckpt_path, data_path, error_show=False):
    '''
    :param ckpt_path: 模型参数文件
    :param data_path: 数据地址
    :param error_show: 是否保存预测错误的图片
    :return:
    '''
    error_path = '/Users/sunyihuan/PycharmProjects/hairstyle/hair_length/error_simple_test'
    if not os.path.exists(error_path):
        os.makedirs(error_path)

    graph = tf.get_default_graph()
    saver = tf.train.import_meta_graph("{}.meta".format(ckpt_path))
    sess = tf.InteractiveSession(graph=graph)
    saver.restore(sess, ckpt_path)
    predict_op = graph.get_tensor_by_name("classes:0")
    features = graph.get_tensor_by_name("features: 0")
    # training_flag = graph.get_tensor_by_name("Placeholder:0")

    te_features, te_labels, _ = get_record_data(graph, data_path, 'test')
    confusion_mat_ = np.zeros((9, 9))
    num = 0
    for _ in range(50):
        try:
            te_feature_batch2, te_label_batch2 = sess.run([te_features, te_labels])
            te_feature_batch2 = te_feature_batch2.reshape(-1, 224, 224, 3)
            print(te_label_batch2)

            prediction = predict_op.eval(feed_dict={features: te_feature_batch2})
            # prediction = np.argmax(prediction, 1)

            num += len(te_label_batch2)

            confusion_mat = confusion_matrix(y_true=te_label_batch2, y_pred=prediction,
                                             labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            confusion_mat_ = confusion_mat_ + confusion_mat

            if error_show:
                for i in range(len(te_label_batch2)):
                    # print("prediction", prediction)
                    # print(i)
                    if prediction[i] != te_label_batch2[i]:
                        Image.fromarray(np.uint8(te_feature_batch2[i]), 'RGB').save(
                            os.path.join(error_path,
                                         "pre{}-cor{}-{}-{}.jpg".format(prediction[i], te_label_batch2[i], i, _)))

        except tf.errors.OutOfRangeError:
            print(_)
    print(confusion_mat_)
    print("accuaracy:", np.trace(np.array(confusion_mat_)) / num)


if __name__ == "__main__":
    ckpt_path = "/Users/sunyihuan/PycharmProjects/myself/DeltaLab/transfer_learning/ckpt/train/model.ckpt-40"
    data_path = "/Users/sunyihuan/Desktop/Data/hair_length/second_data_for_model/complex_all"
    valid_result_analysis(ckpt_path, data_path)
