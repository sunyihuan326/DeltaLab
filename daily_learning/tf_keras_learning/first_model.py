# coding:utf-8 
'''
created on 2018/9/20

@author:sunyihuan
'''
import tensorflow as tf
from tensorflow import keras

# 建立序贯模型
model = keras.Sequential()
# 添加全连接层，节点数为64，激活函数为relu函数，dense表示标准的一维全连接层
model.add(keras.layers.Dense(64, activation='relu'))
# 添加全连接层，节点数为64，激活函数为relu函数
model.add(keras.layers.Dense(64, activation='relu'))
# 添加输出层，输出节点数为10
model.add(keras.layers.Dense(10, activation='softmax'))
# 配置均方误差回归模型
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',  # 均方差
              metrics=['mae'])  # 平均绝对误差

# 配置分类模型
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,  # 多类的对数损失
              metrics=[keras.metrics.categorical_accuracy])  # 多分类问题，所有预测值上的平均正确率

import numpy as np

# 输入数据（1000，32）
data = np.random.random((1000, 32))
# 输入标签（1000，10）
labels = np.random.random((1000, 10))
# 模型训练
model.fit(data, labels, epochs=10, batch_size=32)
#
# # 数据实例化
# dataset = tf.data.Dataset.from_tensor_slices((data, labels))
# dataset = dataset.batch(32)
# dataset = dataset.repeat()
#
# # 模型训练，steps_per_epoch表示每次训练的数据大小类似与batch_size
# model.fit(dataset, epochs=10, steps_per_epoch=30)

# 输入参数
inputs = keras.Input(shape=(32,))

# 网络层的构建
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
# 预测
predictions = keras.layers.Dense(10, activation='softmax')(x)

# 模型实例化
model = keras.Model(inputs=inputs, outputs=predictions)

# 模型构建
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 模型训练
model.fit(data, labels, batch_size=32, epochs=5)


class MyModel(keras.Model):
    # 模型结构确定
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        # 网络层的定义
        self.dense_1 = keras.layers.Dense(32, activation='relu')
        self.dense_2 = keras.layers.Dense(num_classes, activation='sigmoid')

    # 参数调用
    def call(self, inputs):
        # 前向传播过程确定
        x = self.dense_1(inputs)
        return self.dense_2(x)

    def compute_output_shape(self, input_shape):
        # 输出参数确定
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)


# 模型初始化
model = MyModel(num_classes=10)

# 模型构建
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 模型训练
model.fit(data, labels, batch_size=32, epochs=5)




