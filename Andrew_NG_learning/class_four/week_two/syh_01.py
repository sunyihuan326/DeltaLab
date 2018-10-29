# coding:utf-8
'''
Created on 2017/11/23

@author: sunyihuan
'''
import keras
import scipy
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
# import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from class_four.week_two.kt_utils import *
import keras.backend as K

K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

# print("number of training examples = " + str(X_train.shape[0]))
# print("number of test examples = " + str(X_test.shape[0]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
# print("X_test shape: " + str(X_test.shape))
# print("Y_test shape: " + str(Y_test.shape))


def HappyModel(input_shape):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bno')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


happyModel = HappyModel((64, 64, 3))
# # happyModel.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
#                    loss='binary_crossentropy', metrics=['accuracy'])
# happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)
# preds = happyModel.evaluate(x=X_test, y=Y_test)

# print("The Accuracy" + str(preds[1]))
img_path = 'images/panjie0.jpg'
### END CODE HERE ###
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))
# happyModel.summary()
