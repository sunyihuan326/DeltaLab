# coding:utf-8 
'''
created on 2018/3/2

@author:Dxq
'''
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

TYPE = 'test'
source_dir = 'C:/Users/chk01/Desktop/eyelid/test/new'
folders = os.listdir(source_dir)
classes = []
labels = []
images_path = []
for folder in folders:
    classes.append(folder)
    path_folder = os.path.join(source_dir, folder)
    for root, sub_folder, files in os.walk(path_folder):
        for file in files:
            image_path = source_dir + "/" + folder + "/" + file
            images_path.append(image_path)
            labels.append(folder)

label_dict = dict(zip(classes, range(len(classes))))
print(label_dict)
X = np.zeros([len(labels), 64, 64, 3])
Label = []
for i in range(len(labels)):
    img = np.array(Image.open(images_path[i]))
    # dx, dy, dz = np.gradient(img)
    # print(dx.shape)
    # print(dy.shape)
    label = label_dict[labels[i]]
    # if i == 10:
    #     Image.fromarray(img, mode='RGB').show()
    X[i] = img

    Label.append(label)

permutation = list(np.random.permutation(len(labels)))
shuffled_X = X[permutation]
shuffled_Y = [Label[l] for l in permutation]

file = {
    'train': 'model2/data/eyelid_64x64x3_train1796.mat',
    'test': 'model2/data/eyelid_64x64x3_test119.mat',
    'valid': 'model2/data/eyelid_64x64x3_valid102.mat'
}
scio.savemat('test', {"X": shuffled_X, "label": np.array(shuffled_Y).reshape([len(labels), 1])})

data = scio.loadmat('test')
img = data['X'][10]
Image.fromarray(np.uint8(img), mode='RGB').show()

print(img.shape)
# print(data['X'].shape)
# print(data['label'])
