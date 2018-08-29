# -*- coding: UTF-8 -*-
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Arrow, Circle
import dlib
import sys, os
import numpy as np
from shutil import copyfile

FRONT_FACE_STANDARD = "Andre_Agassi_0010.jpg"
shapePredict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
NOSE_CENTER_NUMBER = 30
FRONT_THRESHOLD_DISTANCE = 30
DEST_PATH = "D:\\cvDB\\lfw_30"

def getFaceDis(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    if (len(dets) != 1):
        print "Face number is {0} for {1}, detect failed".format(len(dets), path);
        return None
    detect = dets[0]
    if (detect.left() < 0):
        print "Face detect left is minus for {0}".format(path)
        return None
    if (detect.right() < 0):
        print "Face detect right is minus for {0}".format(path)
        return None
    if (detect.top() < 0):
        print "Face detect top is minus for {0}".format(path)
        return None
    if (detect.bottom() < 0):
        print "Face detect bottom minus for {0}".format(path)
        return None
    shape = shapePredict(img, detect)
    landmarkList = np.zeros((shape.num_parts, 2))
    noseCenter = shape.part(NOSE_CENTER_NUMBER)
    for i in range(shape.num_parts):
        landmarkList[i] = [shape.part(i).x - noseCenter.x, shape.part(i).y - noseCenter.y]
    return landmarkList
    
def getStandardFace():
    img = cv2.cvtColor(cv2.imread(FRONT_FACE_STANDARD), cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)
    shape = shapePredict(img, dets[0])
    landmarkList = np.zeros((shape.num_parts, 2))
    noseCenter = shape.part(NOSE_CENTER_NUMBER)
    for i in range(shape.num_parts):
        landmarkList[i] = [shape.part(i).x - noseCenter.x, shape.part(i).y - noseCenter.y]
    return landmarkList


folder = sys.argv[1]
ext = '.jpg'
standardLandmarks = getStandardFace()
if (not os.path.exists(DEST_PATH)):
    os.mkdir(DEST_PATH)
for file in os.listdir(folder): 
    if (os.path.exists(os.path.join(DEST_PATH, file))):
        continue
    path = os.path.join(folder, file) 
    if os.path.isdir(path): 
        continue
    if not path.endswith(ext):
        continue
    landmarks = getFaceDis(path)
    if landmarks is None:
        continue
    diff = np.linalg.norm(landmarks - standardLandmarks)
    print "{0} distance is {1}".format(path, diff)
    if (diff < FRONT_THRESHOLD_DISTANCE):
        copyfile(path, os.path.join(DEST_PATH, file))
    