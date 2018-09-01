# coding:utf-8 
'''
created on 2018/8/31

@author:sunyihuan
'''
import imutils
import numpy as np
import argparse
import cv2
import imageio

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

# camera = cv2.VideoCapture('/home/yoyomoon/Data/MJ.mp4')
# camera = cv2.imread("/home/yoyomoon/Data/result006.jpg")
# n = camera.isOpened() # return False
# reader = imageio.get_reader('<video0>')

while True:
    # frame = reader.get_next_data()
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.imread("/Users/sunyihuan/Desktop/tt/tt_pic/5b3b0a477c1d0205f8b8dc2f.jpg")
    frame = imutils.resize(frame, width=400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)
    print(skinMask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    cv2.imshow("images", np.hstack([frame, skin]))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cv2.destroyAllWindows()
