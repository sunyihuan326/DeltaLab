# coding:utf-8
"""
Created on 2017.12.27
@author: gcz
"""

from PIL import Image, ImageChops, ImageFile
import time
import os
import urllib
# from urllib2 import urlopen
import urllib2
import StringIO
import cStringIO
import config
mdb = config.mdb
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

# 生成头部图片
def generate_head(front_hair_img, back_hair_img, face_img, forehead_point, chin_point, three_court, scale=1):

    head = back_hair_img
    face_img = resize_image(face_img, FACE_SCALE)
    face_x = HEAD_WIDTH / 2 - int(forehead_point[0]/FACE_SCALE) + 1
    forehead_point_y = int(forehead_point[1]/FACE_SCALE)
    chin_point_y = int(chin_point[1]/FACE_SCALE)
    chin_point_x = int(chin_point[0]/FACE_SCALE)

    forehead_offset = 0
    if three_court == 0:
        # 三庭过长
        forehead_offset = -17
    elif three_court == 1:
        # 三庭过短
        forehead_offset = 12

    head_back = Image.new('RGBA', (HEAD_WIDTH, HEAD_HEIGHT), (0, 0, 0, 0))
    head_back.paste(front_hair_img, (0, forehead_offset))

    face_y = FACE_OFFSET - forehead_point_y
    head.alpha_composite(face_img, (face_x, face_y))
    head.alpha_composite(head_back, (0, 0))

    # 下巴点在头部图片中的位置
    chin_y_offset = chin_point_y + FACE_OFFSET
    chin_x_offset = face_x + chin_point_x
    # 生成缩小的图片给全身图片使用
    if scale != 1:
        head = resize_image(head, scale)
        chin_y_offset = int((chin_point_y + FACE_OFFSET) / scale)
        chin_x_offset = int(chin_x_offset / scale)
    return head, (chin_x_offset, chin_y_offset)

# 缩小图片
def resize_image(image, scale):
    width = int(image.size[0]/scale)
    height = int(image.size[1]/scale)
    i = image.copy()
    i.thumbnail((width, height),Image.ANTIALIAS)
    return i

def scale_image(image, scale):
    width = int(image.size[0]/scale)
    height = int(image.size[1]/scale)
    out = image.resize((width, height), Image.ANTIALIAS)
    return out

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
def whole_pic(body_img, head_img, chin_point):
    x = int(WHOLE_BODY_NECK_POINT[0] - chin_point[0])
    y = int(WHOLE_BODY_NECK_POINT[1] - chin_point[1])
    body_img.alpha_composite(head_img, (x + 3, y))
    return body_img

# 添加背景输出最后图片
def final_pic(hair_img, offset=0, back_url='static/img/back.png'):
    back_ground_pic = Image.open(back_url)
    back = generate_back_img((PIC_WIDTH, PIC_HEIGHT))
    back.paste(hair_img, (0, offset), mask=hair_img)
    back = Image.alpha_composite(back_ground_pic, back)
    back.thumbnail((576, 1024), Image.ANTIALIAS)
    return back

def generate_back_img(img_size):
    return Image.new('RGBA', img_size, (254, 253, 242, 0))

def get_two_whole_body(pic, pic2, offset=0):
    # 生成左边图片
    back = generate_back_img((PIC_WIDTH, PIC_HEIGHT))
    back.paste(pic, (TWO_BODY_LEFT_OFFSET, 0), mask=pic)
    back.paste(pic2, (TWO_BODY_RIGHT_OFFSET, 0), mask=pic2)
    final = Image.open('static/img/back.png')
    final.paste(back, (0, offset), mask=back)
    final.thumbnail((576, 1024), Image.ANTIALIAS)
    return final

def get_recourse(face_img_path, front_hair_path, back_hair_path, body_path):
    # face = get_img_data(face_img_path)
    # front_hair = get_img_data(front_hair_path)
    # back_hair = get_img_data(back_hair_path)
    # body = get_img_data(body_path)

    # if not os.path.exists(face_img_path):
    #     # url = config.srv['upyun']['domain'] + '/xiaomei/customer_face/' + face_img_path.split('/')[-1]
    #     download_img(url)
    # if not os.path.exists(front_hair_path):
    #     # url = config.srv['upyun']['domain'] + '/xiaomei/material_library/video_hair/' + front_hair_path.split('/')[-1]
    #     download_img(url)
    # if not os.path.exists(back_hair_path):
    #     # url = config.srv['upyun']['domain'] + '/xiaomei/material_library/video_hair/' + back_hair_path.split('/')[-1]
    #     download_img(url)
    # if not os.path.exists(back_hair_path):
    #     # url = config.srv['upyun']['domain'] + '/xiaomei/material_library/video_body/' + body_path.split('/')[-1]
    #     download_img(url)

    # _face_path = download_img(face_img_path)
    # _front_path = download_img(front_hair_path)
    # _back_path = download_img(back_hair_path)
    # _body_path = download_img(body_path)

    face = open_file(face_img_path)
    front_hair = open_file(front_hair_path)
    back_hair = open_file(back_hair_path)
    body = open_file(body_path)

    # front_hair = Image.open(front_hair_path)
    # back_hair = Image.open(back_hair_path)
    # body = Image.open(body_path)
    return face, front_hair, back_hair, body

def open_file(url):
    opener = urllib2.build_opener()
    try:
        image_content = opener.open(url).read()
    except Exception as identifier:
        image_content = opener.open(url).read()
        
    image_content = StringIO.StringIO(image_content)
    img = Image.open(image_content)
    # img = None
    # for i in range(10):
    #     try:
    #         img = Image.open(path)
    #         # img.close()
    #     except Exception as identifier:
    #         img = None
    #     if img:
    #         break
    #     time.sleep(0.001)
    # if img == None:
    #     opener = urllib2.build_opener()
    #     image_content = opener.open(url).read()
    #     image_content = StringIO.StringIO(image_content)
    #     img = Image.open(image_content)
    #     # img.close()
    #     print '图片打开失败,下载图片-------------------', path
    #     return img
    return img

def get_img(_filter):
    hair = mdb.xm_hair_img.find_one(_filter)
    hair_img = config.srv['upyun']['domain'] + hair.get('img')
    # hair_img = download_img(hair_img)
    return hair_img

def download_img(url):
    local_path = 'static/img/temp/'
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    url_key = url.split('/')[-1]
    whole_path = local_path + url_key

    if not os.path.exists(whole_path):
        # 本地没有图片下载并保存
        opener = urllib2.build_opener()
        image_content = opener.open(url).read()
        opener.close()
        f = open(whole_path,'wb')
        f.write(image_content)
        f.close()
        image = Image.open(whole_path)
        return whole_path
    else:
        return whole_path

def get_img_data(path):
    data = urllib2.urlopen(path).read()
    data = StringIO.StringIO(data)
    return data

def get_face_param(dic, is_whole=1):
    face_img_path = dic['face_img_path']
    three_court = dic['three_court']
    forehead_point = dic['forehead_point']
    chin_point = dic['chin_point']
    body_path = dic['whole_body'] if is_whole == 1 else dic['half_body']
    return face_img_path, three_court, forehead_point, chin_point, body_path

def get_hair_param(dic):
    front_hair = dic['front_hair_path']
    back_hair = dic['back_hair_path']
    return front_hair, back_hair


def upload_img(pic, plan_id, index, scheme_id=""):
    local_path = 'static/img/' + 'plan_' + str(plan_id) + '_' + str(index) + '.jpg'
    up_key = '/xiaomei/scheme/video/' + str(index) + "_" + str(plan_id) + '_' + str(scheme_id) + '.jpg'
    pic = pic.convert('RGB')
    pic.save(local_path)
    with open(local_path, 'rb') as f:
        try:
            config.upFile.put(up_key, f)
        except Exception as e:
            config.upFile.put(up_key, f)
    try:
        os.remove(local_path)
    except OSError as e:
        pass
    return up_key

# ---------------------------------------------------------

# 获取半身图片
def get_half_body(hair_paths, face_param, offset=0):
    face_img_path, three_court, forehead_point, chin_point, body_path = get_face_param(face_param, is_whole=0)
    front_hair_path, back_hair_path = get_hair_param(hair_paths)
    face, front_hair, back_hair, body = get_recourse(face_img_path, front_hair_path, back_hair_path, body_path)
    head, chin_point = generate_head(front_hair, back_hair, face, forehead_point, chin_point, three_court)
    body = get_body(body)
    pic = half_pic(body, head, chin_point, offset)
    # face.close()
    # front_hair.close()
    # back_hair.close()
    # body.close()
    return pic

# 获取两个半身图片
def get_two_half_body(hair_paths, face_param, big_pic):
    face_img_path, three_court, forehead_point, chin_point, body_path = get_face_param(face_param, is_whole=0)
    front_hair_path, back_hair_path = get_hair_param(hair_paths)
    face, front_hair, back_hair, body = get_recourse(face_img_path, front_hair_path, back_hair_path, body_path)
    head, chin_point = generate_head(front_hair, back_hair, face, forehead_point, chin_point, three_court)
    body = get_body(body)
    pic = half_pic(body, head, chin_point)
    pic = resize_image(pic, 3.5)
    x = 730 + (258-pic.size[0])/2
    y = 395 + (227-pic.size[1])/2
    big_pic.paste(pic,(x, y), mask=pic)
    return big_pic

# 获取半身缩小图片
def get_scale_half_body(hair_paths, face_param, scale=1, offset=0):
    face_img_path, three_court, forehead_point, chin_point, body_path = get_face_param(face_param, is_whole=0)
    front_hair_path, back_hair_path = get_hair_param(hair_paths)
    face, front_hair, back_hair, body = get_recourse(face_img_path, front_hair_path, back_hair_path, body_path)
    head, chin_point = generate_head(front_hair, back_hair, face, forehead_point, chin_point, three_court)
    body = get_body(body)
    pic = half_pic(body, head, chin_point)
    if scale < 1:
        pic = scale_image(pic, scale)
    elif scale > 1:
        pic = resize_image(pic, scale)
    back = generate_back_img((PIC_WIDTH, PIC_HEIGHT))
    x = (PIC_WIDTH - pic.size[0]) / 2
    y = (PIC_HEIGHT- pic.size[1]) / 2
    back.paste(pic, (x, y+offset), mask=pic)
    # face.close()
    # front_hair.close()g
    # back_hair.close()
    # body.close()
    return back

# 获取两个半身图片
def get_two_half_image(original_hair_paths, hair_paths, face_img_path):
    # 设计后的图片
    pic = get_half_body(hair_paths, face_img_path)
    # 原始的图片
    img = get_two_half_body(original_hair_paths, face_img_path, pic)
    return img

# 获取两个全身图片
def get_two_whole_image(original_hair_paths, hair_paths, face_param):
    pic = get_whole_body(original_hair_paths, face_param)
    pic2 = get_whole_body(hair_paths, face_param)
    final = get_two_whole_body(pic, pic2)
    return final

# 获取全身图片
def get_whole_body(hair_paths, face_param):
    face_img_path, three_court, forehead_point, chin_point, body_path = get_face_param(face_param)
    front_hair_path, back_hair_path = get_hair_param(hair_paths)
    face, front_hair, back_hair, body = get_recourse(face_img_path, front_hair_path, back_hair_path, body_path)
    head, chin_point = generate_head(front_hair, back_hair, face, forehead_point, chin_point, three_court, scale=2)
    body = get_body(body)
    pic = whole_pic(body, head, chin_point)
    # face.close()
    # front_hair.close()
    # back_hair.close()
    # body.close()
    return pic

def get_recommend_image(images, plan_id):

    if len(images) != 3:
        images.append(images[0])
        images.append(images[0])

    recommend_back = Image.open('static/img/recommend_back.png')
    img1 = Image.open(images[0])
    img2 = Image.open(images[1])
    img3 = Image.open(images[2])

    img1 = img1.resize((300, 380), Image.ANTIALIAS)
    img2 = img2.resize((300, 380), Image.ANTIALIAS)
    img3 = img3.resize((485, 630), Image.ANTIALIAS)

    back = generate_back_img((1080, 1920))
    back.paste(img1, (120, 510))
    back.paste(img2, (120, 1000))
    back.paste(img3, (480, 610))
    back.paste(recommend_back, (0, 0), mask=recommend_back)
    back.thumbnail((576, 1024), Image.ANTIALIAS)
    return upload_img(back, plan_id, 'recommend')
    # return False
