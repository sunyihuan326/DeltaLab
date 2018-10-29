# coding:utf-8 
'''
created on 2018/3/6

@author:sunyihuan
'''
from keras.models import Sequential
from keras.layers import Dense
import scipy.io as scio
from sklearn.model_selection import train_test_split


def load_data(file, test_size=0.25):
    '''
    :param file:the name of dataset
    :param test_size:float, int, None, optional
                If float, should be between 0.0 and 1.0 and represent the proportion
                of the dataset to include in the test split. If int, represents the
                absolute number of test samples.
    :return: X_train, X_test, Y_train, Y_test shape:[m,features]--[m,classes]
    '''
    data_train = scio.loadmat(file)

    X_train, X_test, Y_train, Y_test = train_test_split(data_train['X'], data_train['Y'], test_size=test_size,
                                                        shuffle=True)

    return X_train, X_test, Y_train, Y_test


model = Sequential()
model.add(Dense(units=64, activation="relu", input_dim=4096))
model.add(Dense(units=3, activation="softmax"))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
file = "/Users/sunyihuan/PycharmProjects/DeltaLab/Sense_model/data/face_1_channel_XY64_sense.mat"
X_train, X_test, Y_train, Y_test = load_data(file, test_size=0.2)
model.fit(X_train, Y_train, epochs=2000, batch_size=32)
loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=128)
print(loss_and_metrics)
# classes = model.predict(X_test, batch_size=128)
