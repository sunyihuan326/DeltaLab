# coding:utf-8
"""
Created on Date
@author: gcz
"""

from PIL import Image, ImageChops
import time

FACE_SCALE = 1.20
FACE_OFFSET = 106
HEAD_WIDTH = 550
HEAD_HEIGHT = 790
FRONT_HAIR_OFFSET = 20
PIC_WIDTH = 1080
PIC_HEIGHT = 1920
HALF_BODY_NECK_POINT = (540, 850)
WHOLE_BODY_NECK_POINT = (475, 552)
SMALL_HALF_PIC_POINT = (675, 165)

TWO_BODY_LEFT_OFFSET = -130
TWO_BODY_RIGHT_OFFSET = 260


def generate_head(front_hair_img, back_hair_img, face_img, forehead_point, chin_point, three_court, scale=1):
    head = back_hair_img
    face_img = resize_image(face_img, FACE_SCALE)
    face_x = HEAD_WIDTH / 2 - int(forehead_point[0] / FACE_SCALE) + 1
    forehead_point_y = int(forehead_point[1] / FACE_SCALE)

    chin_point_y = int(chin_point[1] / FACE_SCALE)
    chin_point_x = int(chin_point[0] / FACE_SCALE)

    forehead_offset = 0
    if three_court == 0:
        # 三庭过长
        forehead_offset = -17
    elif three_court == 1:
        # 三庭过短
        forehead_offset = 12

    head_back = Image.new('RGBA', (HEAD_WIDTH, HEAD_HEIGHT), (0, 0, 0, 0))
    # head_back.paste(front_hair_img, (0, forehead_offset))

    face_y = FACE_OFFSET - forehead_point_y
    head.alpha_composite(face_img, (int(face_x), int(face_y)))
    # head.alpha_composite(head_back, (0, 0))

    # 下巴点在头部图片中的位置
    chin_y_offset = chin_point_y + FACE_OFFSET - forehead_point[1] / scale
    chin_x_offset = face_x + chin_point_x
    # 生成缩小的图片给全身图片使用
    if scale != 1:
        head = resize_image(head, scale)
        chin_y_offset = int((chin_point_y + FACE_OFFSET) / scale)
        chin_x_offset = int(chin_x_offset / scale)
    return head, (chin_x_offset, chin_y_offset)


# 缩小图片
def resize_image(image, scale):
    width = int(image.size[0] / scale)
    height = int(image.size[1] / scale)
    i = image.copy()
    i.thumbnail((width, height), Image.ANTIALIAS)
    return i


# 获得身体图片
def get_body(body_img):
    body_back = generate_back_img((PIC_WIDTH, PIC_HEIGHT))
    return Image.alpha_composite(body_img, body_back)


# 合成半身整体图片不带背景
def half_pic(half_body_img, head_img, chin_point, offset=0):
    x = int(HALF_BODY_NECK_POINT[0] - chin_point[0])
    y = int(HALF_BODY_NECK_POINT[1] - chin_point[1])
    half_body_img.alpha_composite(head_img, (x + 4, y))
    return half_body_img


# 合成全身图片不带背景
def whole_pic(body_img, head_img, chin_offset):
    x = int(WHOLE_BODY_NECK_POINT[0] - head_img.size[0] / 2.0)
    y = int(WHOLE_BODY_NECK_POINT[1] - chin_offset)
    # body_img.paste(head_img, (x, y), mask=head_img)
    body_img.alpha_composite(head_img, (x, y))
    return body_img


# 添加背景输出最后图片
# def final_pic(back_ground_pic, hair_img, offset=0):
#     back = generate_back_img((PIC_WIDTH, PIC_HEIGHT))
#     back.paste(hair_img, (0, offset), mask=hair_img)
#     return Image.alpha_composite(back_ground_pic, back)

def generate_back_img(img_size):
    return Image.new('RGBA', img_size, (254, 253, 242, 0))


def Pic():
    face = Image.open('face.png')

    front_hair = Image.open('中暖_03.png')
    back_hair = Image.open('中暖_03.png')

    # 头 和 头发
    head, chin_point = generate_head(front_hair, back_hair, face, (172, 23), (173, 437), -1)
    # 172,23)173，437
    body = Image.open('body2.png')
    body = get_body(body)
    # 合成头和身体
    pic = half_pic(body, head, chin_point)
    pic.show()
    pic.save('res2.png')
    return pic


def get_scale_half_body(face_img_path, front_hair_path, back_hair_path, body_path, forehead_point, chin_point, scale=1,
                        offset=0):
    face, front_hair, back_hair, body = get_recourse(face_img_path, front_hair_path, back_hair_path, body_path)
    head, chin_offset = generate_head(front_hair, back_hair, face, forehead_point, chin_point)
    body = get_body(body)
    pic = half_pic(body, head, chin_offset)
    pic = resize_image(pic, scale)
    back = generate_back_img((PIC_WIDTH, PIC_HEIGHT))
    x = (PIC_WIDTH - pic.size[0]) / 2
    y = (PIC_HEIGHT - pic.size[1]) / 2
    back.paste(pic, (x, y), mask=pic)
    pic = final_pic(back, offset)
    return pic


def get_recourse(face_img_path, front_hair_path, back_hair_path, body_path):
    face = Image.open(face_img_path)
    front_hair = Image.open(front_hair_path)
    back_hair = Image.open(back_hair_path)
    body = Image.open(body_path)
    return face, front_hair, back_hair, body


def final_pic(hair_img, offset=0, back_url='static/img/back.png'):
    back_ground_pic = Image.open(back_url)
    back = generate_back_img((PIC_WIDTH, PIC_HEIGHT))
    back.paste(hair_img, (0, offset), mask=hair_img)
    return Image.alpha_composite(back_ground_pic, back)


def s_image(image, width, height):
    # width = int(image.size[0]/scale)
    # height = int(image.size[1]/scale)
    i = image.resize((width, height), Image.ANTIALIAS)
    # i = image.copy()
    # i.thumbnail((width, height),Image.ANTIALIAS)
    return i


def open_file(path):
    img = None
    for i in range(10):
        try:
            img = Image.open(path)
        except Exception as identifier:
            img = ''
        if img:
            break
        time.sleep(0.001)
    return img


if __name__ == '__main__':
    Pic()
