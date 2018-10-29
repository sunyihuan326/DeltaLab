# coding:utf-8
'''
Created on 2017/12/27.

@author: chk01
'''
from PIL import Image

img = Image.open('1.png')
img.thumbnail([400, 400])
imgback = Image.new('RGBA', (479, 790))
imgback.paste(img, box=[87, 75])

img1 = Image.open('3.png')
img2 = Image.open('2.png')

res = Image.alpha_composite(imgback, img1)

res = Image.alpha_composite(res, img2)
res.show()
res2 = res.copy()
res2.thumbnail([300, 300])

back = Image.open('back.png')
back.paste(res, mask=res)
back.paste(res2, box=[500, 500], mask=res2)
back.show()
back.save('res.png')
