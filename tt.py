# coding:utf-8
'''
Created on 2018/1/12.

@author: chk01
'''

from PIL import Image, ImageDraw, ImageEnhance

import os


def get_face_box(points):
    X = points[:, 0]
    Y = points[:, 1]
    min_x = min(X) - 10
    max_x = max(X) + 10
    min_y = min(Y) - 10
    max_y = max(Y) + 10

    wid = max(max_y - min_y, max_x - min_x)

    new_x = min_x - (wid - (max_x - min_x)) // 2
    new_y = min_y - (wid - (max_y - min_y)) // 2

    pil_image = Image.new("RGB", (2000, 2000), color=255)
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
    region = region.resize((64, 64), Image.ANTIALIAS)
    region = ImageEnhance.Contrast(region).enhance(999)

    return region


def timeConversion(s):
    if s[-2] == "P":
        if int(s.split(":")[0]) >= 12:
            t__ = s[:-2]
        elif int(s.split(":")[0]) < 12:
            t__ = str(int(s.split(":")[0]) + 12) + s[2:-2]
    elif s[-2] == "A":
        if int(s.split(":")[0]) >= 12:
            a = int(s.split(":")[0]) - 12
            if a < 10:
                a = str(0) + str(a)
            else:
                a = str(a)
            t__ = a + s[2:-2]
        elif int(s.split(":")[0]) < 12:
            t__ = s[:-2]
    return t__
    #
    # Write your code here.
    #


def superReducedString(s):
    s = list(s)
    ll = len(s)
    print(set(s))
    aa = []
    for i in range(ll - 3):
        if s[i + 2] != s[i + 1] and s[i + 1] == s[i]:
            aa.append(s[i + 1])
            aa.append(s[i])
    tt = list(set(s) - set(aa))
    if len(tt) == 0:
        t = "Empty String"
    else:
        t = tt
    return t


def money(a, pv, y):
    '''

    :param a: 每月交付金额
    :param pv: 现有银行利率
    :param y: 要交的年数
    :return:
    '''
    money = 0.0
    for i in range(y * 12):
        k = a * pow((1 + pv / 1200), y * 12 - i)
        money = money + k

    return money


if __name__ == '__main__':
    # s = "aaabccddd"
    #
    # result = superReducedString(s)
    # print(result)
    print(money(983, 3, 20))
    # file = "/Users/sunyihuan/Desktop/1.zip"
    # with open(file):
    #     print("OK")
