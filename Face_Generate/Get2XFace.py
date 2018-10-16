# -*- coding: utf-8 -*- 
import os, sys
import cv2
import numpy as np
from PIL import Image
import dlib
import MovingLSQ as MLSQ
import math
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

folder = sys.argv[1]
savePath = "D:\\cvDB\\lfw\\lfw_out"

facePath = "Andre_Agassi_0010.jpg"
standardImg = cv2.cvtColor(cv2.imread(facePath), cv2.COLOR_BGR2RGB)
detector = dlib.get_frontal_face_detector()
shapePredict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
standardDets = detector(standardImg, 1)[0]
standardShape = shapePredict(standardImg, standardDets)
stanXmin, _, stanYmin, _ = getBound(standardImg, standardShape)
controlDstPts = np.zeros((standardShape.num_parts,2))
for i in range(standardShape.num_parts):
    controlDstPts[i] = [standardShape.part(i).x - stanXmin, standardShape.part(i).y - stanYmin]

def getFaceImg(file):
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    if (len(dets) == 0):
        print("file %s has no face" % file)
        return None
    det = dets[0]
    shape = shapePredict(img, det)
    xmin, xmax, ymin, ymax = getBound(img, shape)
    controlSrcPts = np.zeros((shape.num_parts,2))
    if standardShape.num_parts != shape.num_parts:
        print("number not equal for %s" % file)
        return None
    for i in range(shape.num_parts):
        if (shape.part(i).x < 0):
            print("%d th part x < 0 for %s" % (i, file))
            return None
        if (shape.part(i).y < 0):
            print("%d th part y < 0 for %s" % (i, file))
            return None
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
    FaceX2 = np.zeros((deformedImage.shape[0]*2, deformedImage.shape[1]*2, deformedImage.shape[2]))
    for i in range(len(FaceX2)):
        for j in range(len(FaceX2[0])):
            FaceX2[i][j] = deformedImage[int(i/2)][int(j/2)]
    fullSize = 256
    if FaceX2.shape[0] > fullSize:
        FaceX2 = FaceX2[:fullSize,:,:]
    if FaceX2.shape[1] > fullSize:
        FaceX2 = FaceX2[:,:fullSize,:]
    fullsizeImg = np.zeros((fullSize, fullSize, 3))
    left = int((fullSize - FaceX2.shape[1])/2)
    top = int((fullSize - FaceX2.shape[0])/2)
    fullsizeImg[top:FaceX2.shape[0]+top, left:FaceX2.shape[1]+left, :] = FaceX2
    im = Image.fromarray(fullsizeImg.astype('uint8'))
    im.save(os.path.join(savePath, file[file.rfind('\\') + 1:]))

def processPath(path):
    fileList = os.listdir(path)
    if len(fileList) < 2:
        print("folder %s only has %d files" % (path, len(fileList)))
        return
    for nameFolder in fileList:
        fullPath = os.path.join(path, nameFolder)
        if not os.path.isdir(fullPath):
            continue
        for subPath in os.listdir(fullPath):
            if not subPath.endswith(".jpg"):
                continue
            print("processing %s" % subPath)
            if not os.path.exists(os.path.join(savePath, nameFolder)):
                os.mkdir(os.path.join(savePath, nameFolder))
            if os.path.exists(os.path.join(savePath, nameFolder, subPath)):
                continue
            else:
                try:
                    getFaceImg(os.path.join(fullPath, subPath))
                except Exception as err:
                    print("err " + err)


processPath(folder)