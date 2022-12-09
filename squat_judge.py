import natsort
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import shutil

from deep_squat_model import build_model
from plot_hist import plot_hist
from deep_squat_cosine_decay import WarmupCosineDecay
from deep_squat_train import train_squat_classifier
from barbell_tracker import barbell_tracker_train, barbell_tracker_detect

train_classifier = True
train_tracker = True
video_path = 'C:/Users/Ran/Documents/ATG_AI/ATG.AI/IMG_0441.mov'

train_squat_classifier(train_classifier)

barbell_tracker_train(train_tracker)

barbell_tracker_detect(video_path)



labels_dir = 'yolov5/runs/detect/exp/labels'
frames = os.listdir(labels_dir)
frames = natsort.natsorted(frames)