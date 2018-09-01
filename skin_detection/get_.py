# coding:utf-8 
'''
created on 2018/8/31

@author:sunyihuan
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

################################################################################

print('Pixel Values Access')

imgFile = '/Users/sunyihuan/Desktop/tt/tt_pic/5b3b0a477c1d0205f8b8dc2f.jpg'

# load an original image
img = cv2.imread(imgFile)

# access a pixel at (row,column) coordinates
px = img[150, 200]
print('Pixel Value at (150,200):', px)

# access a pixel from blue channel
blue = img[150, 200, 0]
# access a pixel from green channel
green = img[150, 200, 1]
# access a pixel from red channel
red = img[150, 200, 2]
print('Pixel Value from B,G,R channels at (150,200): ', blue, green, red)
################################################################################

print('Pixel Values Modification')

img[150, 200] = [0, 0, 0]
print('Modified Pixel Value at (150,200):', px)
################################################################################
# better way: using numpy

# access a pixel from blue channel
blue = img.item(100, 200, 0)
# access a pixel from green channel
green = img.item(100, 200, 1)
# access a pixel from red channel
red = img.item(100, 200, 2)
print('Pixel Value using Numpy from B,G,R channels at (100,200): ', blue, green, red)

# warning: we can only change pixels in gray or single-channel image

# modify green value: (row,col,channel)
img.itemset((100, 200, 1), 255)
# read green value
green = img.item(100, 200, 1)
print('Modified Green Channel Value Using Numpy at (100,200):', green)
################################################################################

print('Skin Model')

rows, cols, channels = img.shape

# prepare an empty image space
imgSkin = np.zeros(img.shape, np.uint8)
# copy original image
imgSkin = img.copy()

for r in range(rows):
    for c in range(cols):

        # get pixel value
        B = img.item(r, c, 0)
        G = img.item(r, c, 1)
        R = img.item(r, c, 2)

        # non-skin area if skin equals 0, skin area otherwise
        skin = 0

        if (abs(R - G) > 15) and (R > G) and (R > B):
            if (R > 95) and (G > 40) and (B > 20) and (max(R, G, B) - min(R, G, B) > 15):
                skin = 1
                # print 'Condition 1 satisfied!'
            elif (R > 220) and (G > 210) and (B > 170):
                skin = 1
                # print 'Condition 2 satisfied!'

        if 0 == skin:
            imgSkin.itemset((r, c, 0), 0)
            imgSkin.itemset((r, c, 1), 0)
            imgSkin.itemset((r, c, 2), 0)
            # print 'Skin detected!'

# convert color space of images because of the display difference between cv2 and matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgSkin = cv2.cvtColor(imgSkin, cv2.COLOR_BGR2RGB)

# display original image and skin image
plt.subplot(1, 2, 1), plt.imshow(img), plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(imgSkin), plt.title('Skin Image'), plt.xticks([]), plt.yticks([])
plt.show()
################################################################################

print('Waiting for Key Operation')

k = cv2.waitKey(0)

# wait for ESC key to exit
if 27 == k:
    cv2.destroyAllWindows()
