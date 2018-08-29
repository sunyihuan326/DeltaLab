# coding:utf-8 
'''
created on 2018/8/23

@author:sunyihuan
'''

import grpc
import numpy as np
from PIL import Image
import threading
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc, prediction_service_pb2

import time
from serving_learning.hair_length_service import corp_region, get_baseInfo_tx, preprocess

server = "47.93.235.220:8502"

file_name = "/Users/sunyihuan/Desktop/Data/check_data/hair_length/0815fachang-yan_check0/tapai/2/8929.jpg"
land_point = get_baseInfo_tx(file_name)
im = corp_region(file_name, land_point)
im = preprocess(np.array(im.resize([224, 224])).reshape([-1, 224, 224, 3]))
start = time.time()

channel = grpc.insecure_channel(server)

stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = "inceptionV4_hair_len"
request.model_spec.signature_name = 'predict_images'

# put data into TensorProto and copy them into the request object

request.inputs["images"].CopyFrom(
    tf.contrib.util.make_tensor_proto(im.astype(dtype=np.float32), shape=[1, 224, 224, 3]))

result_future = stub.Predict.future(request, 10.0)
result = result_future.result().outputs['scores'].int64_val
end = time.time()
print(result)
print(end - start)
