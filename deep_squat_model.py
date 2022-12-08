import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
import shutil


# load dataset labels
labels_path = 'Shallow_Squat_Error_Dataset/labels_shallow_depth.json'
labels = pd.read_json(labels_path,  typ='dictionary', convert_axes=False ,convert_dates=False)
labels = labels.to_frame()
labels = labels.reset_index()

# load training, validation and test split IDs
ids_path = 'Shallow_Squat_Error_Dataset/splits'
train_ids_path =  ids_path + '/train_ids.json'
val_ids_path = ids_path + '/val_ids.json'
test_ids_path = ids_path + '/test_ids.json'

train_ids = pd.read_json(train_ids_path, dtype=False, convert_axes=False ,convert_dates=False)
val_ids = pd.read_json(val_ids_path, dtype=False, convert_axes=False ,convert_dates=False)
test_ids = pd.read_json(test_ids_path, dtype=False, convert_axes=False ,convert_dates=False)

# Create train, val, test directories
parent_dir = 'Shallow_Squat_Error_Dataset'
src_dir = 'Shallow_Squat_Error_Dataset/crops_unaligned'
train_img_path = os.path.join(parent_dir, 'Train')
train_img_shallow = os.path.join(train_img_path, 'shallow')
train_img_deep = os.path.join(train_img_path, 'deep')
val_img_path = os.path.join(parent_dir, 'Valid')
val_img_shallow = os.path.join(val_img_path, 'shallow')
val_img_deep = os.path.join(val_img_path, 'deep')
test_img_path = os.path.join(parent_dir, 'Test')
test_img_deep = os.path.join(test_img_path, 'deep')
test_img_shallow = os.path.join(test_img_path, 'shallow')

try:
    os.makedirs(train_img_path)
    os.makedirs(train_img_shallow)
    os.makedirs(train_img_deep)
    os.makedirs(val_img_path)
    os.makedirs(val_img_shallow)
    os.makedirs(val_img_deep)
    os.makedirs(test_img_path)
    os.makedirs(test_img_deep)
    os.makedirs(test_img_shallow)
except:
    print('Folders already exist')

# Copy images to the appropriate folders but only if they exist in the train/val/test split
filenames = os.listdir(src_dir)

for file in filenames:
  try:
    file_label = labels.loc[labels["index"] == file[:-4]].values[0][1]
  except:
    continue

  if file[:-4] in train_ids.values:
    if file_label:
      shutil.copy(os.path.join(src_dir, file), train_img_deep)
    else:
      shutil.copy(os.path.join(src_dir, file), train_img_shallow)
  if file[:-4] in val_ids.values:
    if file_label:
      shutil.copy(os.path.join(src_dir, file), val_img_deep)
    else:
      shutil.copy(os.path.join(src_dir, file), val_img_shallow)
  if file[:-4] in test_ids.values:
    if file_label:
      shutil.copy(os.path.join(src_dir, file), test_img_deep)
    else:
      shutil.copy(os.path.join(src_dir, file), test_img_shallow)

# Get number of training examples
num_train = 0
filenames = os.listdir(train_img_shallow)
for file in filenames:
  num_train += 1
filenames = os.listdir(train_img_deep)
for file in filenames:
  num_train += 1

# Define image size and batch size
batch_size = 32
img_height = 299
img_width = 299

# Create image data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=0.0,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                horizontal_flip=True)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=0.0,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2,
                                                                horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_img_path, 
                                                    target_size=(img_height, img_width),
                                                    color_mode="rgb",
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    shuffle=True,
                                                    )
val_generator = train_datagen.flow_from_directory(val_img_path, 
                                                    target_size=(img_height, img_width),
                                                    color_mode="rgb",
                                                    batch_size=32,
                                                    class_mode="categorical",
                                                    shuffle=True)
