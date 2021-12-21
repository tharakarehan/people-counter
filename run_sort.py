import os
import time
import random

import cv2
import imutils
#import dlib
import numpy as np
#import tensorflow as tf
import argparse
from utils import image_utils, model_utils
from utils import Comparator

from sort import Sort

#Initialize tracker
# entry = 0
# exit = 0


parser = argparse.ArgumentParser(description='Run SORT')
parser.add_argument('--input_file', type=str, help='Input videos file path name')
parser.add_argument('--output_file', type=str, help='Output video file path name')
parser.add_argument('--model_path', type=str, help='path to the model')
parser.add_argument('--threshold', type=float, help='threshold for detections')
args = parser.parse_args()


model="tensorflow_hub"
tracker = Sort(use_dlib=False)



# initialize the video stream, pointer to output video file, and frame dimensions
inputFile=args.input_file
vs = cv2.VideoCapture(inputFile)
fps = int(vs.get(cv2.CAP_PROP_FPS))
total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
(W, H) = (int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT)))

result = cv2.VideoWriter(args.output_file,  
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         fps, (W,H))

Tr = args.threshold
# get line info

# line = image_utils.define_ROI(input_file, H, W)


if model=='Haar':
    person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    frame_index = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        detections = model_utils.get_haar_detections(frame, person_cascade, frame_index)
        trackers = tracker.update(detections, frame)

        for d in trackers:
            d = d.astype(np.int32)

            frame = image_utils.draw_box(frame, d, (0,255,0))

            #if detections != []:
                #cv2.putText(frame, 'Detection active', (W-10,H-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        result.write(frame)
        frame_index += 1

    result.release()
    vs.release()

elif model=='hog':

    frame_index = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        detections = model_utils.get_hog_svm_detections(frame, frame_index)
        trackers = tracker.update(detections, frame)

        for d in trackers:
            d = d.astype(np.int32)

            frame = image_utils.draw_box(frame, d, (0,255,0))

            #if detections != []:
                #cv2.putText(frame, 'Detection active', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        result.write(frame)
        frame_index += 1

    result.release()
    vs.release()

elif model=='tensorflow':

    frame_index = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        detections = model_utils.get_tensorflow_detections(frame)
        trackers = tracker.update(detections, frame)

        for d in trackers:
            d = d.astype(np.int32)

            frame = image_utils.draw_box(frame, d, (0,255,0))

            #if detections != []:
                #cv2.putText(frame, 'Detection active', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        result.write(frame)
        frame_index += 1

    result.release()
    vs.release()

elif model=='pedestron':
    Model = model_utils.initialize_pedestron()
    frame_index = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        detections = model_utils.get_pedestron_detection(Model,frame,thresh=0.7)
        trackers = tracker.update(detections, frame)

        current={}
        for d in trackers:
            d = d.astype(np.int32)

            frame = image_utils.draw_box(frame, d, (0,255,0))

            if detections != []:
                cv2.putText(frame, 'Detection active', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            current[d[4]] = (d[0], d[1], d[2], d[3])
            if d[4] in tracker.previous:
                previous_box = tracker.previous[d[4]]
                entry, exit = Comparator.compare_with_prev_position(previous_box, d, line, entry, exit)

        tracker.previous = current
        frame = image_utils.annotate_frame(frame, line, entry, exit, H, W)
        cv2.imshow('pedestron',frame)
        cv2.waitKey(1)
        result.write(frame)
        frame_index += 1

    result.release()
    vs.release()

elif model=='tensorflow_hub':
    Model = model_utils.initialize_tensorflow_hub(args.model_path)
    frame_index = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        detections = model_utils.get_tensorflow_detections(Model,frame,Tr,W,H)
        trackers = tracker.update(detections, frame)

        for d in trackers:
            d = d.astype(np.int32)

            frame = image_utils.draw_box(frame, d, (0,255,0))

            #if detections != []:
                #cv2.putText(frame, 'Detection active', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        result.write(frame)
        frame_index += 1
        if frame_index%fps==0: print(int(frame_index/fps),'seconds_completed')

    result.release()
    vs.release()

