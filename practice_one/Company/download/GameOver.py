# coding:utf-8
'''
Created on 2018/1/24.

@author: chk01
'''
import copy
from aip import AipFace
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import scipy.io as scio
import requests
from io import BytesIO
import base64
import tensorflow as tf
import xlsxwriter
import os
import time

# "圆形0", "心型1", "三角2", "鹅蛋脸3", "方形脸4"
SHAPE_TRANS = {'oval': "A", 'heart': "B", 'square': "C", 'triangle': "D", 'round': "E"}
SHAPE_NAME = {"A": "鹅蛋脸", "B": "心型脸", "C": "方脸", "D": "三角脸", "E": "圆脸"}
SHAPE_NUM = {"A": 3, "B": 1, "C": 4, "D": 2, "E": 0}
STYLE_TRANS = ['甜美可爱', '自然优雅', '浪漫迷人', '魅力时尚', '高雅柔美', '华丽高雅', '清纯简洁', '知性沉着', '现代摩登']
DOMAIN = 'https://devai.meiyezhushou.com'
COURT = ['上庭过长', '上庭过短', '正常']


class Face(object):
    def __init__(self, file, stature=0, sense=0, age=35):
        '''
        :param file: local img path
        '''
        # stature"苗条0", "适中1", "丰腴2"
        # sense量感大0 正常1
        # age18-30-0， 31-40-1 40+-2
        self.org_file = file
        self.image = Image.open(file)
        self.width = self.image.size[0]
        self.stature = stature
        self.sense = sense
        self.age = age
        self.client = AipFace('10365287', 'G7q4m36Yic1vpFCl5t46yH5K', 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC')
        self.__get_baidu_info()
        self.__get_organ_struct()
        self.__get_faceshape()
        self.feature = {}
        self.top = ''

    def get_face_top(self):
        self.__load_top_model()
        image = Image.open(self.org_file).convert("L")
        region, wid = self.__get_top_region()

        new_x = region[0]
        new_y = region[1]
        trX = np.array(image.crop(region).resize([64, 64]))
        top = np.squeeze(self.predict_op.eval({self.trX: trX.reshape(1, -1) / 255.})).reshape(-1, 2) * wid / 64 + \
              [new_x, new_y]
        self.top = [int(top[1][0]), int(top[1][1])]
        return top

    def get_cartoon_face(self):
        res = requests.get(
            DOMAIN + '/cartoon/merge?landmark72={}&&glasses={}&&faceshape={}&&skin_color={}'.format(self.landmark72,
                                                                                                    self.glasses,
                                                                                                    self.baidu_faceshape,
                                                                                                    1),
            cert=False).json()

        imgdata = base64.b64decode(str(res['cartoon_face'])[2:-1])
        img = Image.open(BytesIO(imgdata))
        self.feature = res['feature_index']

        return img

    def get_style(self):
        self.__load_style_model()
        trX = np.array(self.__get_face_box()).reshape(1, -1)

        sense = self.__get_sense64(trX / 255.)
        outline = self.__get_outline64(trX / 255.)
        sense_2 = self.__get_sense64_2classes(trX / 255.)
        outline_2 = self.__get_outline64_2classes(trX / 255.)
        outline_merge = outline if (outline - outline_2 * 2) == 0 else 1
        sense_merge = sense if (sense - sense_2 * 2) == 0 else 1
        style = 3 * outline_merge + sense_merge
        return self.__expert_check(style)

    def baidu_check(self):
        '''
        :return:PIL.Image object
        '''
        im = self.image
        draw = ImageDraw.Draw(im)
        landmark72 = tuple(tuple(t) for t in self.formed_landmark72)
        draw.line(landmark72[:13], fill=255, width=int(self.width * 0.005))
        self.image = im
        return im

    def top_check(self):
        im = self.image
        draw = ImageDraw.Draw(im)
        top = tuple(tuple(t) for t in self.get_face_top())
        draw.line(top, fill=(255, 255, 255), width=2)
        self.image = im
        return im

    def struct_check(self):
        '''
        :return:PIL.Image object
        '''
        im = self.image
        draw = ImageDraw.Draw(im)
        # p1 = tuple(tuple(t) for t in self.organ_struct)
        # draw.line(p1[:5], fill=0, width=1)
        # draw.line((p1[-2], p1[-1], p1[2]), fill=0, width=1)

        p2 = tuple(tuple(t) for t in self.formed_organ_struct)
        draw.line(p2[:5], fill=(255, 255, 255), width=2)
        draw.line((p2[-2], p2[-1], p2[2]), fill=(255, 255, 255), width=2)
        draw.line(p2[2:4], fill=(0, 255, 0), width=2)
        draw.point(p2, fill=(0, 0, 0))
        self.image = im

        return im

    def three_court_check(self):
        if not self.top:
            self.get_face_top()

        top = self.top
        brow = (self.formed_landmark72[26] + self.formed_landmark72[39]) / 2
        nose = (self.formed_landmark72[51] + self.formed_landmark72[52]) / 2
        chin = self.formed_landmark72[6]

        ab = int(abs(top[1] - brow[1]))
        bc = int(abs(brow[1] - nose[1]))
        cd = int(abs(nose[1] - chin[1]))
        center = chin[0] + 30
        k = 270.0 / (bc + cd)
        ab *= k
        bc *= k
        cd *= k

        if (ab > bc) and (ab > cd):
            three_court = 0
        elif (ab < bc) and (ab < cd):
            three_court = 1
        else:
            three_court = 2
        self.up_len = int(ab)
        self.mid_len = int(bc)
        self.down_len = int(cd)
        self.three_court = three_court

        im = self.image
        draw = ImageDraw.Draw(im)
        points = [[center, top[1]], [center, brow[1]], [center, nose[1]], [center, chin[1]]]
        p2 = tuple(tuple(t) for t in points)
        draw.line(p2[0:2], fill=(255, 0, 0), width=2)
        draw.line(p2[1:3], fill=(255, 255, 0), width=2)
        draw.line(p2[2:4], fill=(0, 0, 255), width=2)
        self.image = im
        return im

    def report(self):
        file = self.org_file
        xlsx_file = file.split('.')[0] + '.xlsx'
        book = xlsxwriter.Workbook(xlsx_file)
        sheet = book.add_worksheet('report')
        self.baidu_check()
        self.struct_check()
        self.top_check()
        self.three_court_check()

        im = self.image.convert("RGB")
        im.save('report.jpg')
        cartoon = self.get_cartoon_face()
        cartoon.save("cartoon.png")
        sheet.insert_image('N2', 'cartoon.png')
        scale = 500.0 / self.width
        sheet.insert_image('A2', 'report.jpg', options={"x_scale": scale, "y_scale": scale})
        feature = self.feature

        sheet.write_string("L2", '眉毛')
        sheet.write_number("M2", feature['left_eyebrow'])
        sheet.write_string("L3", '眼睛')
        sheet.write_number("M3", feature['left_eye'])
        sheet.write_string("L4", '鼻子')
        sheet.write_number("M4", feature['nose'])
        sheet.write_string("L5", '嘴巴')
        sheet.write_number("M5", feature['lip'])
        sheet.write_string("L6", '脸型')
        sheet.write_string("M6", feature['chin'])

        sheet.write_string("I9", '上庭')
        sheet.write_number("J9", self.up_len)
        sheet.write_string("I10", '中庭')
        sheet.write_number("J10", self.mid_len)
        sheet.write_string("I11", '下庭')
        sheet.write_number("J11", self.down_len)
        sheet.write_string("I12", '结果')
        sheet.write_string("J12", COURT[self.three_court])
        book.close()
        os.remove('report.jpg')
        os.remove('cartoon.png')

    def __get_top_region(self):
        points = self.formed_landmark72
        X = points[:, 0]
        Y = points[:, 1]
        min_x = min(X)
        max_x = max(X)
        min_y = min(Y)
        max_y = max(Y)
        wid = max(max_y - min_y, max_x - min_x)
        wid = 1.8 * wid
        new_x = min_x - (wid - (max_x - min_x)) // 2
        new_y = min_y - (wid - (max_y - min_y)) // 2
        p = 0.2
        region = [int(new_x), int(new_y - p * wid), int(new_x + wid), int(new_y + (1 - p) * wid)]
        return region, wid

    def __expert_check(self, style):
        ml = 2 * self.stature + self.sense
        if self.age < 26:
            _age = 0
        elif self.age < 35:
            _age = 1
        else:
            _age = 2

        af = 5 * _age + SHAPE_NUM[self.faceshape]

        return self.expert_exp[style][ml][af]

    def __get_outline64(self, trX):
        W = self.outline_parameters['W1']
        b = self.outline_parameters['b1']
        Z = np.add(np.matmul(trX, W.T), b)
        return np.squeeze(np.argmax(Z, 1))

    def __get_outline64_2classes(self, trX):
        W = self.outline_2classes_parameters['W1']
        b = self.outline_2classes_parameters['b1']
        Z = np.add(np.matmul(trX, W.T), b)
        return np.squeeze(np.argmax(Z, 1))

    def __get_sense64(self, trX):
        W = self.sense_parameters['W1']
        b = self.sense_parameters['b1']
        Z = np.add(np.matmul(trX, W.T), b)
        return np.squeeze(np.argmax(Z, 1))

    def __get_sense64_2classes(self, trX):
        W = self.sense_2classes_parameters['W1']
        b = self.sense_2classes_parameters['b1']
        Z = np.add(np.matmul(trX, W.T), b)
        return np.squeeze(np.argmax(Z, 1))

    def __get_face_box(self):
        points = self.formed_landmark72
        X = points[:, 0]
        Y = points[:, 1]
        min_x = min(X) - 10
        max_x = max(X) + 10
        min_y = min(Y) - 10
        max_y = max(Y) + 10

        wid = max(max_y - min_y, max_x - min_x)

        new_x = min_x - (wid - (max_x - min_x)) // 2
        new_y = min_y - (wid - (max_y - min_y)) // 2

        pil_image = Image.new("RGB", (2000, 2000), color=(255, 255, 255))
        d = ImageDraw.Draw(pil_image)

        d.line([tuple(p) for p in points[:13]], width=10, fill=0)
        d.line([tuple(p) for p in points[13:21]], width=10, fill=0)
        d.line([tuple(p) for p in [points[13], points[20]]], width=10, fill=0)
        d.line([tuple(p) for p in points[30:38]], width=10, fill=0)
        d.line([tuple(p) for p in [points[30], points[37]]], width=10, fill=0)
        d.line([tuple(p) for p in points[22:30]], width=10, fill=0)
        d.line([tuple(p) for p in [points[22], points[29]]], width=10, fill=0)
        d.line([tuple(p) for p in points[39:47]], width=10, fill=0)
        d.line([tuple(p) for p in [points[39], points[46]]], width=10, fill=0)
        d.line([tuple(p) for p in points[47:57]], width=10, fill=0)
        d.line([tuple(p) for p in points[58:66]], width=10, fill=0)
        d.line([tuple(p) for p in [points[58], points[65]]], width=10, fill=0)

        region = pil_image.crop([new_x, new_y, new_x + wid, new_y + wid])
        region = region.resize((64, 64), Image.ANTIALIAS).convert("L")
        region = ImageEnhance.Contrast(region).enhance(999)
        return region

    def __get_faceshape(self):
        faceshape = self.baidu_faceshape
        _faceshape = sorted(faceshape, key=lambda x: -x['probability'])
        default_shape = _faceshape[0]['type']

        feature_dict = {}
        for i in range(len(faceshape)):
            feature_dict.update({faceshape[i]['type']: faceshape[i]['probability']})

        if feature_dict['triangle'] > 0.45:
            new_shape = 'D'
        elif feature_dict['oval'] > 0.30:
            new_shape = 'A'
        elif feature_dict['heart'] > 0.60:
            new_shape = 'B'
        elif feature_dict['square'] > 0.20:
            new_shape = 'C'
        elif feature_dict['round'] > 0.15:
            new_shape = 'E'
        else:
            new_shape = SHAPE_TRANS[default_shape]
        self.faceshape = new_shape
        # self.faceshape = SHAPE_NAME[new_shape]

    def __get_organ_struct(self):
        landmark72 = self.formed_landmark72
        ors = [(landmark72[24] + landmark72[28]) / 2, landmark72[21],
               (landmark72[51] + landmark72[52]) / 2, landmark72[60],
               landmark72[6], (landmark72[41] + landmark72[45]) / 2,
               landmark72[38]]
        self.organ_struct = ors
        self.formed_organ_struct = self.__org_alignment(ors)

    def __org_alignment(self, ors):
        new_ors = copy.deepcopy(ors)
        eb_y = (new_ors[0][1] + new_ors[-2][1]) / 2
        eye_y = (new_ors[1][1] + new_ors[-1][1]) / 2
        eye_dis = (new_ors[-1][0] - new_ors[1][0]) / 2
        eb_dis = (new_ors[-2][0] - new_ors[0][0]) / 2
        new_ors[0][1] = eb_y
        new_ors[-2][1] = eb_y
        new_ors[1][1] = eye_y
        new_ors[-1][1] = eye_y
        center_x = (new_ors[2][0] + new_ors[3][0] + new_ors[4][0]) / 3
        new_ors[2][0] = center_x
        new_ors[3][0] = center_x
        new_ors[4][0] = center_x

        new_ors[1][0] = center_x - eye_dis
        new_ors[-1][0] = center_x + eye_dis

        new_ors[0][0] = center_x - eb_dis
        new_ors[-2][0] = center_x + eb_dis
        return new_ors

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
        self.baidu_faceshape = result['faceshape']

    def __load_style_model(self):
        self.outline_parameters = scio.loadmat('material/style/outline3.mat')
        self.outline_2classes_parameters = scio.loadmat('material/style/outline2.mat')
        self.sense_parameters = scio.loadmat('material/style/sense3.mat')
        self.sense_2classes_parameters = scio.loadmat('material/style/sense2.mat')
        self.expert_exp = scio.loadmat('material/style/style_expert_exp-2.mat')['data']

    def __load_top_model(self):
        tf.reset_default_graph()
        saver = tf.train.import_meta_graph("material/face/model.ckpt.meta")
        sess = tf.InteractiveSession()
        saver.restore(sess, "material/face/model.ckpt")
        graph = tf.get_default_graph()
        self.predict_op = graph.get_tensor_by_name("output/BiasAdd:0")
        self.trX = graph.get_tensor_by_name("Placeholder:0")


ts = time.time()
face = Face('check/1.jpg', stature=0, sense=0, age=35)
face.report()
print(time.time() - ts)
# im = face.get_cartoon_face()
# im.show()
# im = face.baidu_check()
# im2 = face.get_organ_struct()
# im2.save('1.jpg')
# im2.show()
