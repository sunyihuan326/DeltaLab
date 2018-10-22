# coding:utf-8 
'''
created on 2018/10/17

@author:sunyihuan
'''
import shutil
import os


def picture_copy(file_root_dir, file_search, file_query):
    '''
    将目标图和被搜到的图放到一个文件夹中
    :param file_root_dir: 分好的图要保存的根目录
    :param file_search_dir: 目标图
    :param file_query: 被选中的图
    :return:
    '''
    f = file_search.split("/")[-1]  # 获取目标图片的名字
    to_save_dir = os.path.join(file_root_dir, str(f.split(".")[0]))  # 要保存的文件夹

    os.makedirs(to_save_dir, exist_ok=True)  # 创建文件夹

    file_query_name = file_query.split("/")[-1]  # 获取被选中图片的名字

    shutil.copy(file_search, os.path.join(to_save_dir, f))
    shutil.copy(file_query, os.path.join(to_save_dir, file_query_name))
