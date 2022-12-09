import natsort
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import shutil

from deep_squat_train import train_squat_classifier
from barbell_tracker import barbell_tracker_train, barbell_tracker_detect

'''
I have already trained the classifier and tracker models for you.
However, you can retrain them should you like to.
Just change the following flags to True.
'''

train_classifier = False #change to True if you want to train the deep squat image classifier
classifier_inference = False #change to True if you want to run classification on images in the test folder
train_tracker = False #change to True if you want to train the yolov5 custom barbell tracker

video_path = 'C:/Users/Ran/Documents/ATG_AI/ATG.AI/IMG_0441.mov'

train_squat_classifier(train_classifier, classifier_inference)

if train_tracker:
    barbell_tracker_train(train_tracker)

barbell_tracker_detect(video_path)



labels_dir = 'yolov5/runs/detect/exp/labels'
frames = os.listdir(labels_dir)
frames = natsort.natsorted(frames)