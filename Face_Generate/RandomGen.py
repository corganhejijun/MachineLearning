# -*- coding: utf-8 -*- 
import os, sys
import cv2
import numpy as np
from PIL import Image
import dlib

folder = sys.argv[1]
savePath = "D:\\git\\MachineLearning\\Face_Generate\\out"
detector = dlib.get_frontal_face_detector()
shapePredict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def getBound(img, shape):
    xMin = len(img[0])
    xMax = 0
    yMin = len(img)
    yMax = 0
    for i in range(shape.num_parts):
        if (shape.part(i).x < xMin):
            xMin = shape.part(i).x
        if (shape.part(i).x > xMax):
            xMax = shape.part(i).x
        if (shape.part(i).y < yMin):
            yMin = shape.part(i).y
        if (shape.part(i).y > yMax):
            yMax = shape.part(i).y
    return xMin, xMax, yMin, yMax

def trueOrFalse():
    return np.random.random_sample() > 0.5

def plusOrMinus():
    number = np.random.random_sample()
    if (number < 0.3):
        return -1
    if (number > 0.7):
        return 1
    return 0

def getFace(img):
    dets = detector(img, 1)
    if (len(dets) == 0):
        print("file %s has no face" % file)
        return None, None, None, None
    det = dets[0]
    shape = shapePredict(img, det)
    return getBound(img, shape)

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

def resizeImg(path, noMask = False):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    if (noMask):
        maskImg = img
    else:
        maskImg = genRandomMask(img)
        xmin, xmax, ymin, ymax = getFace(img)
        if not xmin:
            return None
        maskImg[:,:xmin] = [0, 0, 0]
        maskImg[:,xmax:] = [0, 0, 0]
        maskImg[:ymin,:] = [0, 0, 0]
        maskImg[ymax:,:] = [0, 0, 0]

    target = Image.new('RGB', (img.shape[0]*2, img.shape[1]))
    target.paste(Image.fromarray(img), (0, 0))
    target.paste(Image.fromarray(maskImg), (img.shape[0] + 1, 0))
    return target

for file in os.listdir(folder): 
    if not file.endswith(".jpg"):
        continue
    path = os.path.join(folder, file) 
    print("processing %s" % (path))
    result = resizeImg(path, False)
    if not result:
        continue
    resultPath = os.path.join(savePath, file)
    result.save(resultPath)
    print("saved to %s" % resultPath)