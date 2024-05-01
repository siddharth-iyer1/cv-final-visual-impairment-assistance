import ultralytics
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import time
import math
    
def captureImages():
    cap = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    frames = []
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    ret, frame = cap2.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        
    cap.release()
    cap2.release()
    return frames


def getObjInFrame(frame, model):
    results = model(frame)
    boxes = results[0].boxes
    obj_list = []
    for box in boxes:
        obj_pred = results[0].names[int(box.cls[0].item())]
        obj_coords = box.xyxy[0].tolist()
        pair = (obj_pred, calcCenter(obj_coords))
        obj_list.append(pair)
    
    annotated_frame = results[0].plot()
    annotated_frame_cv = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)
    cv2.imshow("Images", annotated_frame_cv)

    return obj_list


# Calculate the center coordinates given the xyxy coordinates 

def calcCenter(coords):
    center_xy = []
    center_x = coords[0]+coords[2] / 2
    center_y = coords[1]+coords[3] / 2
    center_xy.append(center_x)
    center_xy.append(center_y)
    return center_xy


# Given two lists of box objects detected by YOLO from 2 different images, return a list of pairs of matching objects (object, coords)
def matchObjects(box1, box2):
    matches = []
    for item1 in box1:
        obj_class_1 = item1[0]
        for item2 in box2:
            obj_class_2 = item2[0]
            if(obj_class_1 == obj_class_2):
                match = (item1, item2)
                matches.append(match)
            else:
                continue
    return matches



model = YOLO('yolov8n.pt')
while(1):
    frames = captureImages()
    obj_in_frame1_list = getObjInFrame(frames[0], model=model)
    obj_in_frame2_list = getObjInFrame(frames[1], model=model)
    matches = matchObjects(obj_in_frame1_list, obj_in_frame2_list)
    print(matches)

