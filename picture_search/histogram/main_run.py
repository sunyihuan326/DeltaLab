# coding:utf-8 
'''
created on 2018/10/15

@author:sunyihuan
'''
import os
import numpy as np
from PIL import Image
from picture_search.histogram.histogram import calMultipleHistogramSimilarity  # 直方图距离计算图片相似度
import time


def output_similar_picture(file_search, file_query_dir):
    '''
    输出相似度最高的图片地址
    :param file_search: 目标图片地址
    :param file_query_dir: 被搜索文件夹
    :return:
    '''
    HistogramSimilarity = []
    dir_list = os.listdir(file_query_dir)
    for i, file in enumerate(dir_list):
        if file != ".DS_Store":
            HistogramSimilarityData = calMultipleHistogramSimilarity(file_search, os.path.join(file_query_dir, file))
            HistogramSimilarity.append(HistogramSimilarityData)
    # print(len(HistogramSimilarity))
    simlar_max = np.argmax(HistogramSimilarity)
    # print(simlar_max)
    file_query_file = dir_list[simlar_max]
    return os.path.join(file_query_dir, file_query_file)


file_search = "/Users/sunyihuan/Desktop/tt/65978163b01bdf8e.jpg"
file_query_dir = "/Users/sunyihuan/Desktop/unlike"

start = time.time()
file_query_path = output_similar_picture(file_search, file_query_dir)
end = time.time()
print("all_time:", end - start)
Image.open(file_query_path).show()
