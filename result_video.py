import shutil
import os
import cv2
import tensorflow as tf
import numpy as np
from clear_files import clear_files

def classify_video(classification_model,
                   video_path,
                   peaks,
                   frames_before,
                   frames_after,
                   deep_threshold,
                   img_height=299,
                   img_width=299):
    '''
    Receives the Tensorflow classification model, the path of the frames of the video,
    a list of peak frames, frames to consider before and after each peak,
    deep_threshold fraction of frames that need to be classified as
    deep squats for the squat to count as deep, image height and width.
    Returns a list of whether each peak is a deep or shallow squat.
    '''
    cap = cv2.VideoCapture(video_path)
    folder = 'test/frames/'

    os.makedirs(folder, exist_ok=True)

    deep_squats = []

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    for frame_number in peaks:
        images = []
        for i in range(frame_number-frames_before,
                       frame_number + frames_after + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            res, frame = cap.read()
            images.append(frame)
            filename = f'test/frames/{i}.jpg'
            cv2.imwrite(filename, frame)

        test_generator = test_datagen.flow_from_directory(
            directory='test/',
            target_size=(img_height, img_width),
            color_mode="rgb",
            batch_size=1,
            class_mode="categorical",
            shuffle=False
            )

        squat_class = classification_model.predict(test_generator)
        squat_class_argmax = np.argmax(squat_class, axis=1)

        '''
        # plot classification on frames for debug
        for j, classification in enumerate(squat_class_argmax):
            img = images[j]
            plt.imshow(img[:,:,::-1])
            if squat_class_argmax[j]:
                plt.title('Shallow squat!')
            else:
                plt.title('Deep squat!')
                plt.show()
        '''
        # If enough frames in the window are classified as deep squats
        # we classify the squat as deep
        if len(np.where(squat_class_argmax == 0)[0]) / \
           len(squat_class_argmax) >= deep_threshold:
            deep_squats.append(1)
        else:
            deep_squats.append(0)

        # Remove .jpg files from folder

        clear_files(folder)

    print('Valid squats in video: ', np.sum(deep_squats))
    return deep_squats


def result_video(video_path, result_video_path, peaks, deep_squats):
    '''
    Opens the tracking video and writes a version of it
    with deep/shallow squat written on the relevant frames.
    Saves it to result_video.avi
    '''
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result_vid = cv2.VideoWriter(result_video_path,
                                 fourcc,
                                 30.0,
                                 (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_counter = 0
    i = 0
    write_counter = 0
    # Read until video is completed
    try:
        while cap.isOpened():

            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                height = frame.shape[0]
                width = frame.shape[1]
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                # org
                org = (50, 200)
                # fontScale
                # this value can be from 0 to 1 (0,1] to
                # change the size of the text relative to the image
                scale = 1 
                font_scale = min(width, height)/(250/scale)
                # Blue color in BGR
                color = (255, 0, 0)
                # Line thickness of 2 px
                thickness = 8
                # Using cv2.putText() method
                if i < len(peaks):
                    # pick the window around the identified peak frame
                    if frame_counter >= peaks[i] and \
                    frame_counter <= peaks[i] + 30:
                        if deep_squats[i]:
                            text = 'Deep squat'
                        else:
                            text = 'Shallow squat'
                        frame = cv2.putText(frame, text, org, font,
                                            font_scale, color, thickness,
                                            cv2.LINE_AA)
                        # plt.imshow(frame[:,:,::-1])
                        # plt.show()
                        write_counter += 1

                    if write_counter >= 30:
                        i += 1
                        write_counter = 0

                result_vid.write(frame)
                frame_counter += 1
            else:
                break
    finally:
        # When everything done, release the video capture object
        cap.release()
        result_vid.release()
