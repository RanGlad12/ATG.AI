import natsort
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import scipy as sp
import scipy.signal

from deep_squat_train import train_squat_classifier
from barbell_tracker import barbell_tracker_train, barbell_tracker_detect
from find_tracker_peaks import find_tracker_peaks
from deep_squat_model import build_model
from result_video import classify_video, result_video
from choose_video import choose_video

'''
I have already trained the classifier and tracker models for you.
However, you can retrain them should you like to.
Just change the following flags to True.
Note that to train the classifier you need access to a deep/shallow
squat database, as I cannot provide the database I have used
due to a user agreement and privacy concerns.
'''
# change to True if you want to train
#  the deep squat image classifier
train_classifier = False
# change to True if you want to run
#  classification on images in the test folder
classifier_inference = False
# change to True if you want to
# train the yolov5 custom barbell tracker
train_tracker = False

video_path = choose_video()

if train_classifier:
    train_squat_classifier(classifier_inference)

if train_tracker:
    barbell_tracker_train(train_tracker)

barbell_tracker_detect(video_path)


# order the detected frames
labels_dir = 'yolov5/runs/detect/exp/labels'
frames = os.listdir(labels_dir)
frames = natsort.natsorted(frames)

x, y = find_tracker_peaks(frames, labels_dir)
x = np.asarray(x)
y = np.asarray(y)

# find frames corresponding to the bottom of the squat,
# i.e. barbell is at the lowest point
prominence = 0.03
width = 20
peaks, properties = sp.signal.find_peaks(y, prominence=prominence, width=width)


plt.plot(y)
plt.plot(peaks, y[peaks], "x", markersize=10, linewidth=10)
plt.title('Identify squat frames')
plt.xlabel('Frame number')
plt.ylabel('Relative y coordinate')
plt.show()

checkpoint_path = 'deep_squat.hdf5'
classification_model = build_model(num_classes=2,
                                   img_height=299,
                                   img_width=299)
classification_model.load_weights(checkpoint_path)

# Define frames to classify according to around the peaks
frames_before = 8
frames_after = 0

deep_squats = classify_video(classification_model,
                             video_path,
                             peaks,
                             frames_before, 
                             frames_after,
                             img_height=299,
                             img_width=299)
output_path = 'result_video.avi'
result_video(video_path, output_path, peaks, deep_squats)


# clear folders
def clear_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


clear_files('test/frames')
clear_files('yolov5/runs/detect')

print('Done!')
