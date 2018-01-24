# coding:utf-8
'''
Created on 2018/1/24.

@author: chk01
'''

from aip import AipFace
from PIL import Image, ImageDraw
import numpy as np

SHAPE_TRANS = {'oval': "A", 'heart': "B", 'square': "C", 'triangle': "D", 'round': "E"}


class Face(object):
    def __init__(self, file):
        '''
        :param file: local img path
        '''
        self.org_file = file
        self.client = AipFace('10365287', 'G7q4m36Yic1vpFCl5t46yH5K', 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC')
        self.__get_baidu_info()
        self.__get_organ_struct()

    def get_face_top(self):
        pass

    def get_cartoon_face(self):
        pass

    def get_style(self):
        pass

    def baidu_check(self):
        '''
        :return:PIL.Image object
        '''
        im = Image.open(self.org_file)
        draw = ImageDraw.Draw(im)
        landmark72 = tuple(tuple(t) for t in self.formed_landmark72)
        draw.line(landmark72[:13], fill=255, width=1)
        return im

    def get_organ_struct(self):
        '''
        :return:PIL.Image object
        '''
        im = Image.open(self.org_file)
        draw = ImageDraw.Draw(im)
        pp = tuple(tuple(t) for t in self.organ_struct)
        draw.line(pp[:5], fill=255, width=1)
        draw.line((pp[-2], pp[-1], pp[2]), fill=255, width=1)
        return im

    def __get_organ_struct(self):
        landmark72 = self.formed_landmark72
        self.organ_struct = [(landmark72[24] + landmark72[28]) / 2, landmark72[21],
                             (landmark72[51] + landmark72[52]) / 2, landmark72[60],
                             landmark72[6], (landmark72[41] + landmark72[45]) / 2,
                             landmark72[38]]

    def __landmark72_trans(self, points):
        num = len(points)
        formed_data = np.zeros([num, 2])
        formed_data[:, 0] = [p['x'] for p in points]
        formed_data[:, 1] = [p['y'] for p in points]
        return formed_data

    def __get_file_content(self, filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    def __get_baidu_info(self):
        options = {
            'max_face_num': 2,
            'face_fields': "landmark,faceshape,gender,glasses,qualities"
        }
        res = self.client.detect(self.__get_file_content(self.org_file), options=options)
        if res['result_num'] != 1:
            raise Exception("Two many faces")

        result = res['result'][0]
        if result['face_probability'] < 0.8:
            raise Exception("It might not be a face")
        isPerson = result['qualities']['type']['human']
        isCartoon = result['qualities']['type']['cartoon']

        if isPerson < isCartoon:
            raise Exception("It might be a cartoon face rather than a real person")

        self.landmark72 = result['landmark72']
        self.formed_landmark72 = self.__landmark72_trans(result['landmark72'])
        self.glasses = result['glasses']
        self.faceshape = result['faceshape']

    def __load_style_model(self):
        pass

    def __load_top_model(self):
        pass


face = Face('check/1.jpg')
# im = face.baidu_check()
im2 = face.get_organ_struct()
im2.show()
