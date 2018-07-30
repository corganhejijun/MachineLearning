# -*- coding: utf-8 -*- 
import os, sys
import cv2
import numpy as np
from PIL import Image

folder = sys.argv[1]
savePath = "D:\\cvDB\\frontal_30\\train"

def trueOrFalse():
    return np.random.random_sample() > 0.5

def plusOrMinus():
    number = np.random.random_sample()
    if (number < 0.3):
        return -1
    if (number > 0.7):
        return 1
    return 0

def genRandomMask(img):
    imgArray = np.array(img)
    step = 10
    gap = 3
    # 竖向条纹
    x = 10
    for i in range(int(imgArray.shape[1] / step)):
        if (trueOrFalse()):
            continue
        viber = 0
        for j in range(imgArray.shape[0]):
            viber += plusOrMinus()
            y = i*step + viber
            if y > imgArray.shape[1] - 1 - gap:
                y = imgArray.shape[1] - 1 - gap
            imgArray[j, y:y+gap, :] = [0, 0, 0]
    # 横向条纹
    y = 10
    for i in range(int(imgArray.shape[0] / step)):
        if (trueOrFalse()):
            continue
        for j in range(imgArray.shape[1]):
            viber += plusOrMinus()
            x = i*step + viber
            if x > imgArray.shape[0] - 1 - gap:
                x = imgArray.shape[0] - 1 - gap
            imgArray[x:x+gap, j, :] = [0, 0, 0]
    return imgArray

def resizeImg(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    maskImg = genRandomMask(img)
    target = Image.new('RGB', (img.shape[0]*2, img.shape[1]))
    target.paste(Image.fromarray(img), (0, 0))
    target.paste(Image.fromarray(maskImg), (img.shape[0] + 1, 0))
    return target

for file in os.listdir(folder): 
    if not file.endswith(".jpg"):
        continue
    path = os.path.join(folder, file) 
    print "processing {0}".format(path)
    result = resizeImg(path)
    result.save(os.path.join(savePath, file))