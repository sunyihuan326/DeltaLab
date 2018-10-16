# coding:utf-8 
'''
created on 2018/10/15

@author:sunyihuan
'''
import glob, os
import argparse
import cv2
from picture_search.picture_descriptor import ColorDescriptor

aparse = argparse.ArgumentParser()
aparse.add_argument("-i", "--indexFile", help="Path to where the computed index will be stored")
aparse.add_argument("-d", "--directory", help="Path to the directory that contains the images to be indexed")
args = vars(aparse.parse_args())

cd = ColorDescriptor((8, 12, 3))  # 初始化图片描述

picture_list = glob.glob(args["directory"] + "/*.png")  # search pic
output = open(args["indexFile"], "w")  # 打开索引文件准备写入图片的信息

for pic in picture_list:
    imgID = os.path.basename(pic)
    img = cv2.imread(pic)

    features = cd.describe(img)
    # print features
    features = [str(f) for f in features]
    output.write("%s,%s\n" % (imgID, ",".join(features)))

output.close()
