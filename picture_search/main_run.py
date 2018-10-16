# coding:utf-8 
'''
created on 2018/10/15

@author:sunyihuan
'''

import argparse
import cv2
from picture_search.picture_descriptor import ColorDescriptor
from picture_search.picture_descriptor import Searcher

aparse = argparse.ArgumentParser()
aparse.add_argument("-q", "--queryFile", required=True, help="Path to the query image")
aparse.add_argument("-i", "--indexFile", required=True, help="Path to where the computed index will be stored")
aparse.add_argument("-r", "--result_path", required=True, help="Path to the result path")
args = vars(aparse.parse_args())

cd = ColorDescriptor((8, 12, 3))  # 初始化图片描述
query_img = cv2.imread(args["queryFile"])
features = cd.describe(query_img)  # 获取指定图片的特征

searcher = Searcher(args["indexFile"])
results = searcher.search(features)
print(results)

# cv2.imshow("Query", query_img) #显示指定的图片

for (score, resultID) in results:
    print(score, resultID)
    # 显示搜索到的符合要求的照片
    result = cv2.imread(args["result_path"] + "/" + resultID)
# print result
# cv2.imshow("Result", result)
# cv2.waitKey(0)
