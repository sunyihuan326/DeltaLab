# coding:utf-8
'''
Created on 2017/11/24.

@author: chk01
'''
from PIL import Image, ImageDraw
import numpy as np
from aip import AipFace

""" 你的 APPID AK SK """
APP_ID = '10365287'
API_KEY = 'G7q4m36Yic1vpFCl5t46yH5K'
SECRET_KEY = 'MneS2GDvPQ5QsGpVtSaHXGAlvwHu1XnC '

client = AipFace(APP_ID, API_KEY, SECRET_KEY)


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


def landmark72_trans(points):
    num = len(points)
    data = np.zeros([num, 2])
    data[:, 0] = [p['x'] for p in points]
    data[:, 1] = [p['y'] for p in points]
    return data


def get_landmark72(full_path):
    options = {
        'max_face_num': 1,
        # 'face_fields': "age,beauty,expression,faceshape,gender,glasses,landmark,race,qualities",
        'face_fields': "landmark"
    }
    result = client.detect(get_file_content(full_path), options=options)
    landmark72 = landmark72_trans(result['result'][0]['landmark72'])
    return landmark72


# chin 13# data1 = data[:13]
# eyes# data2 = data[13:22]# data2.extend(data[30:39])
# browns# data3 = data[22:30]# data3.extend(data[39:47])
# nose# data4 = data[47:58]
# mouse# data5 = data[58:]
if __name__ == '__main__':
    file = 'check/6.jpg'

    im = Image.open(file)
    drawSurface = ImageDraw.Draw(im)
    landmark72 = get_landmark72(file)
    landmark72 = tuple(tuple(t) for t in landmark72)
    drawSurface.line(landmark72[:13], fill=255, width=2)
    drawSurface.line(landmark72[13:22], fill=100, width=10)
    # drawSurface.line(landmark72[14], fill=100, width=10)
    drawSurface.line(landmark72[30:39], fill=100, width=10)
    # drawSurface.line(landmark72[34], fill=100, width=10)

    drawSurface.line(landmark72[22:30], fill=150, width=2)
    drawSurface.line(landmark72[39:47], fill=150, width=2)
    drawSurface.line(landmark72[47:58], fill=50, width=2)
    drawSurface.line(landmark72[58:], fill=50, width=2)

    im.show()
