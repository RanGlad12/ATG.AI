import os
import shutil
import natsort
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from deep_squat_train import train_squat_classifier
from barbell_tracker import barbell_tracker_train, barbell_tracker_detect
from get_tracker import get_tracker
from deep_squat_model import build_model
from result_video import classify_video, result_video
from choose_video import choose_video
from clear_files import clear_files


# change to True if you want to train
#  the deep squat image classifier
TRAIN_CLASSIFIER = False
# change to True if you want to run
#  classification on images in the test folder
CLASSIFIER_INFERENCE = False
# change to True if you want to
# train the yolov5 custom barbell tracker
TRAIN_TRACKER = False

if TRAIN_CLASSIFIER:
    train_squat_classifier(CLASSIFIER_INFERENCE)

if TRAIN_TRACKER:
    barbell_tracker_train(TRAIN_TRACKER)

video_path = choose_video()
barbell_tracker_detect(video_path)


# order the detected frames acording to video order
LABELS_DIR = 'yolov5/runs/detect/exp/labels'
frames = os.listdir(LABELS_DIR)
frames = natsort.natsorted(frames)

x, y = get_tracker(frames, LABELS_DIR)
x = np.asarray(x)
y = np.asarray(y)

# find frames corresponding to the bottom of the squat,
# i.e. barbell is at the lowest point
PROMINENCE = 0.03
WIDTH = 20
peaks, properties = sp.signal.find_peaks(y, prominence=PROMINENCE, width=WIDTH)

'''
# Plot peaks for debug purposes
plt.plot(y)
plt.plot(peaks, y[peaks], "x", markersize=10, linewidth=10)
plt.title('Identify squat frames')
plt.xlabel('Frame number')
plt.ylabel('Relative y coordinate')
plt.show()
'''

CHECKPOINT_PATH = 'deep_squat.hdf5'
classification_model = build_model(num_classes=2,
                                   img_height=299,
                                   img_width=299)
classification_model.load_weights(CHECKPOINT_PATH)

# Define frames to classify according to around the peaks
FRAMES_BEFORE = 8
FRAMES_AFTER = 0

deep_squats = classify_video(classification_model,
                             video_path,
                             peaks,
                             FRAMES_BEFORE,
                             FRAMES_AFTER,
                             img_height=299,
                             img_width=299)
OUTPUT_PATH = 'result_video.avi'
video_name = video_path.split('/')
video_name = video_name[-1]
video_name = video_name.split('.')
video_name = video_name[0]
tracking_video_path = 'yolov5/runs/detect/exp/' + video_name + '.mp4'
result_video(tracking_video_path, OUTPUT_PATH, peaks, deep_squats)

clear_files('test/frames')
clear_files('yolov5/runs/detect')

print('Done!')
