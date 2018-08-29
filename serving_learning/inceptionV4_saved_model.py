# coding:utf-8 
'''
created on 2018/8/25

@author:sunyihuan
'''

import tensorflow as tf

import tensorflow_serving

hair_length_graph = tf.Graph()
import os

model_path = "/Users/sunyihuan/Desktop/parameters/hair_length/second/complex77_simple70/model.ckpt-39"
saved_model_path = "/Users/sunyihuan/PycharmProjects/myself/DeltaLab/serving_learning/inceptionV4_saved_model"


def main():
    with hair_length_graph.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(model_path))
        sess = tf.InteractiveSession(graph=hair_length_graph)
        saver.restore(sess, model_path)

        predict_op = hair_length_graph.get_tensor_by_name("classes:0")
        features = hair_length_graph.get_tensor_by_name("features:0")

        builder = tf.saved_model.builder.SavedModelBuilder(
            os.path.join(saved_model_path, str(2)))

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


main()