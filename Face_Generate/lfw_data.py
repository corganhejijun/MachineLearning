# -*- coding: UTF-8 -*-
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Arrow, Circle
import dlib
import math
import sys, os
import numpy as np
from shutil import copyfile
import MovingLSQ as MLSQ
from PIL import Image

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

FRONT_FACE_STANDARD = "Andre_Agassi_0010.jpg"
standardImg = cv2.cvtColor(cv2.imread(FRONT_FACE_STANDARD), cv2.COLOR_BGR2RGB)
shapePredict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
standardDets = detector(standardImg, 1)[0]
standardShape = shapePredict(standardImg, standardDets)
stanXmin, _, stanYmin, _ = getBound(standardImg, standardShape)
controlDstPts = np.zeros((standardShape.num_parts,2))
for i in range(standardShape.num_parts):
    controlDstPts[i] = [standardShape.part(i).x - stanXmin, standardShape.part(i).y - stanYmin]
NOSE_CENTER_NUMBER = 30
FRONT_THRESHOLD_DISTANCE = 30
DEST_PATH = "D:\\cvDB\\lfw\\lfw_data"

def getFaceDis(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    if (len(dets) != 1):
        print("Face number is {0} for {1}, detect failed".format(len(dets), path));
        return None, None, None
    detect = dets[0]
    if (detect.left() < 0):
        print("Face detect left is minus for {0}".format(path))
        return None, None, None
    if (detect.right() < 0):
        print("Face detect right is minus for {0}".format(path))
        return None, None, None
    if (detect.top() < 0):
        print("Face detect top is minus for {0}".format(path))
        return None, None, None
    if (detect.bottom() < 0):
        print("Face detect bottom minus for {0}".format(path))
        return None, None, None
    shape = shapePredict(img, detect)
    landmarkList = np.zeros((shape.num_parts, 2))
    noseCenter = shape.part(NOSE_CENTER_NUMBER)
    for i in range(shape.num_parts):
        landmarkList[i] = [shape.part(i).x - noseCenter.x, shape.part(i).y - noseCenter.y]
    return landmarkList, img, shape

def transform(shape, file, img):
    xmin, xmax, ymin, ymax = getBound(img, shape)
    controlSrcPts = np.zeros((shape.num_parts,2))
    if standardShape.num_parts != shape.num_parts:
        print("number not equal for %s" % file)
        return []
    for i in range(shape.num_parts):
        if (shape.part(i).x < 0):
            print("%d th part x < 0 for %s" % (i, file))
            return []
        if (shape.part(i).y < 0):
            print("%d th part y < 0 for %s" % (i, file))
            return []
        controlSrcPts[i] = [shape.part(i).x - xmin, shape.part(i).y - ymin]
    solver = MLSQ.MovingLSQ(controlSrcPts, controlDstPts)
    imgIdx = np.zeros(((xmax - xmin)*(ymax - ymin), 2))
    for i in range((ymax - ymin)):
        for j in range((xmax - xmin)):
            imgIdx[i*(xmax - xmin) + j] = [j, i]
    imgMls = solver.Run_Rigid(imgIdx)
    mlsMargin = [0, 0, (xmax - xmin), (ymax - ymin)]
    for i in range(len(imgMls)):
        if (imgMls[i][0] < mlsMargin[0]):
            mlsMargin[0] = imgMls[i][0]
        if (imgMls[i][1] < mlsMargin[1]):
            mlsMargin[1] = imgMls[i][1]
        if (imgMls[i][0] > mlsMargin[2]):
            mlsMargin[2] = imgMls[i][0]
        if (imgMls[i][1] > mlsMargin[3]):
            mlsMargin[3] = imgMls[i][1]
    mlsMargin[2] -= (xmax - xmin)
    mlsMargin[3] -= (ymax - ymin)
    imgMlsMap = imgMls.reshape(((ymax - ymin), (xmax - xmin), 2))
    leftMargin = -math.floor(mlsMargin[0])
    topMargin = -math.floor(mlsMargin[1])
    rightMargin = math.ceil(mlsMargin[2])
    bottomMargin = math.ceil(mlsMargin[3])
    deformedImage = np.zeros(((ymax - ymin) + int(topMargin) + int(bottomMargin), 
                                (xmax - xmin) + int(leftMargin) + int(rightMargin), 3))
    cropImg = img[ymin:ymax,xmin:xmax,:]
    for i in range(len(cropImg)):
        for j in range(len(cropImg[0])):
            x = int(math.floor(imgMlsMap[i][j][0]) + leftMargin)
            y = int(math.floor(imgMlsMap[i][j][1]) + topMargin)
            if (x < 0 or y < 0):
                break
            deformedImage[y, x] = cropImg[i, j]
    return deformedImage
    
def getStandardFace():
    landmarkList = np.zeros((standardShape.num_parts, 2))
    noseCenter = standardShape.part(NOSE_CENTER_NUMBER)
    for i in range(standardShape.num_parts):
        landmarkList[i] = [standardShape.part(i).x - noseCenter.x, standardShape.part(i).y - noseCenter.y]
    return landmarkList

def X2(img):
    FaceX2 = np.zeros((img.shape[0]*2, img.shape[1]*2, img.shape[2]))
    for i in range(len(FaceX2)):
        for j in range(len(FaceX2[0])):
            FaceX2[i][j] = img[int(i/2)][int(j/2)]
    fullSize = 256
    if FaceX2.shape[0] > fullSize:
        FaceX2 = FaceX2[:fullSize,:,:]
    if FaceX2.shape[1] > fullSize:
        FaceX2 = FaceX2[:,:fullSize,:]
    fullsizeImg = np.zeros((fullSize, fullSize, 3))
    left = int((fullSize - FaceX2.shape[1])/2)
    top = int((fullSize - FaceX2.shape[0])/2)
    fullsizeImg[top:FaceX2.shape[0]+top, left:FaceX2.shape[1]+left, :] = FaceX2
    return fullsizeImg


folder = sys.argv[1]
ext = '.jpg'
standardLandmarks = getStandardFace()
if (not os.path.exists(DEST_PATH)):
    os.mkdir(DEST_PATH)
for subFolder in os.listdir(folder): 
    subPath = os.path.join(folder, subFolder) 
    if not os.path.isdir(subPath): 
        continue
    imgList = os.listdir(subPath)
    if len(imgList) < 2:
        print("folder {0} has only {1} files".format(subFolder, len(imgList)))
        continue
    print("processing {0} files in {1}".format(len(imgList), subFolder))
    foundFront = False
    for imgFile in imgList:
        imgPath = os.path.join(subPath, imgFile)
        if not imgPath.endswith(ext):
            continue
        landmarks, faceImg, shape = getFaceDis(imgPath)
        if landmarks is None:
            continue
        diff = np.linalg.norm(landmarks - standardLandmarks)
        print("{0} distance is {1}".format(imgFile, diff))
        if (diff < FRONT_THRESHOLD_DISTANCE):
            foundFront = True
    if not foundFront:
        print("Not found front face in {0}".format(subFolder))
        continue
    frontImg = []
    notFrontList = []
    imgFileNameList = []
    for imgFile in imgList:
        imgPath = os.path.join(subPath, imgFile)
        if not imgPath.endswith(ext):
            continue
        landmarks, faceImg, shape = getFaceDis(imgPath)
        if landmarks is None:
            continue
        diff = np.linalg.norm(landmarks - standardLandmarks)
        print("{0} distance is {1}".format(imgFile, diff))
        if (diff < FRONT_THRESHOLD_DISTANCE):
            xmin, xmax, ymin, ymax = getBound(faceImg, shape)
            frontImg = faceImg[ymin:ymax, xmin:xmax, :]
            frontImg = X2(frontImg)
        else:
            if os.path.exists(os.path.join(DEST_PATH, imgFile)):
                continue
            notFrontImg = transform(shape, imgFile, faceImg)
            if len(notFrontImg) > 0:
                notFrontList.append(X2(notFrontImg))
                imgFileNameList.append(imgFile)
    if len(frontImg) == 0 or len(notFrontList) == 0:
        continue
    for index, img in enumerate(notFrontList):
        target = Image.new('RGB', (img.shape[0]*2, img.shape[1]))
        target.paste(Image.fromarray(np.uint8(frontImg)), (0, 0))
        target.paste(Image.fromarray(np.uint8(img)), (img.shape[0] + 1, 0))
        resultPath = os.path.join(DEST_PATH, imgFileNameList[index])
        target.save(resultPath)
        print("saved to %s" % resultPath)
    
    