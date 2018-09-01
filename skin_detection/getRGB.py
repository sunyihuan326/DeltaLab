# coding:utf-8 
'''
created on 2018/8/31

@author:sunyihuan
'''
import colorsys
from sklearn.cluster import KMeans

from PIL import Image
from skin_detection.utils import get_baseInfo_tx
import cv2


def get_face_region(file_path):
    '''
    获取脸部区域
    :param file_path:
    :return:
    '''
    face_data = get_baseInfo_tx(file_path)
    if face_data["roll"] > 5 or face_data["roll"] < -5:
        Image.open(file_path).rotate(-face_data['roll']).save(file_path)  # 图片旋转后保存
        face_data = get_baseInfo_tx(file_path)
    x_left = min(face_data["left_eye"][:, 0])
    x_right = max(face_data["right_eye"][:, 0])
    y_upper = max(face_data["left_eye"][:, 1])
    y_lower = min(face_data["mouth"][:, 1])

    region = (x_left, y_upper, x_right, y_lower)
    im = Image.open(file_path).convert("RGB").crop(region)
    save_path = file_path[:-4] + "_crop" + ".jpg"
    im.save(save_path)
    dominant_color = get_color(im)
    print(dominant_color)

    im = Image.new("RGB", size=(100, 100), color=dominant_color)
    im.show()


def get_color(image):
    '''
    获取颜色（均值、众数）
    :param image:
    :return:
    '''
    image = image.convert('RGBA')
    # image.thumbnail((200, 200))
    R = []
    G = []
    B = []
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        if a == 0:
            continue
        R.append(r)
        G.append(g)
        B.append(b)
    rr = int(sum(R) / len(R))
    gg = int(sum(G) / len(G))
    bb = int(sum(B) / len(B))
    rr_z = get_zhongshu(R)
    gg_z = get_zhongshu(G)
    bb_z = get_zhongshu(B)

    return (int((rr + rr_z) / 2), int((gg + gg_z) / 2), int((bb + bb_z) / 2))


def get_zhongshu(aa):
    '''
    获取数组众数
    :param aa:
    :return:
    '''
    rr_ = dict((a, aa.count(a)) for a in aa)
    vv = max(rr_.values())
    zhongshu = []
    for k in rr_.keys():
        if rr_[k] == vv:
            zhongshu.append(k)
    return zhongshu[0]


def get_dominant_color(image):
    '''
    获取主色调
    :param image:
    :return:
    '''
    # 颜色模式转换，以便输出rgb颜色值
    image = image.convert('RGBA')

    # 生成缩略图，减少计算量，减小cpu压力
    image.thumbnail((200, 200))

    max_score = 0
    dominant_color = 0

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue

        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        #
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)

        y = (y - 16.0) / (235 - 16)

        # 忽略高亮色
        if y > 0.93:
            continue

        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count

        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)

    return dominant_color


def pil_getcolor(imgFile):
    img = cv2.imread(imgFile)
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    kmeans = KMeans(n_clusters=2).fit(r)
    pre = kmeans.cluster_centers_
    print(pre)

    # print(image.getcolors(image.size[0] * image.size[1]))


imgFile = '/Users/sunyihuan/Desktop/tt/tt_pic/5b3b1f577c1d020603824646.jpg'
pil_getcolor(imgFile)
# get_face_region(imgFile)
# image = Image.open(imgFile)

# dominant_color = get_dominant_color(image)
# im = Image.new("RGB", size=(50, 50), color=dominant_color)
# im.show()
