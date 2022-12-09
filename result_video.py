import cv2
import shutil
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np

def classify_video(classification_model, video_path, peaks, img_height=299, img_width=299):
    cap = cv2.VideoCapture(video_path)
    folder = 'test/frames/'

    try:
        os.makedirs(folder)
    except:
        print('directory already exists')

    # define the window around the peak for which frames will be classified
    frames_before = 8 
    frames_after = 0
    #at least deep_threshold of the frames need to be classified as deep for the squat to count as deep
    deep_threshold = 0.3
    deep_squats = []


    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()


    for frame_number in peaks:
        images = []
        for i in range(frame_number-frames_before, frame_number + frames_after + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i )
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
    
        if len(np.where(squat_class_argmax == 0)[0]) / len(squat_class_argmax) >= deep_threshold:
            deep_squats.append(1)
        else:
            deep_squats.append(0)

    
        #Remove .jpg files form folder
            
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    

    print('Valid squats in video: ', np.sum(deep_squats))
    return deep_squats

def result_video(video_path, result_video_path, peaks, deep_squats):
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result_vid = cv2.VideoWriter(result_video_path, fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    #result_vid = cv2.VideoWriter(result_video_path, fourcc, 20.0,(int(cap.get(3)),int(cap.get(4))))


    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    frame_counter = 0 
    i = 0
    write_counter = 0
    # Read until video is completed
    while(cap.isOpened()):


        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # org
            org = (50, 200)
            
            # fontScale
            fontScale = 3.5
            
            # Blue color in BGR
            color = (255, 0, 0)
            
            # Line thickness of 2 px
            thickness = 8
            
            # Using cv2.putText() method
            if i < len(peaks):
                if frame_counter >= peaks[i] and frame_counter <= peaks[i] + 20:
                    if deep_squats[i]:
                        text = 'Deep squat'
                    else:
                        text = 'Shallow squat'
                    frame = cv2.putText(frame, text, org, font, 
                                    fontScale, color, thickness, cv2.LINE_AA)
                    #plt.imshow(frame[:,:,::-1])
                    #plt.show()
                    write_counter += 1

                if write_counter >= 20 - 1:
                    i += 1
                    write_counter = 0
                

            result_vid.write(frame)
            frame_counter += 1
        else:
            break

        
    # When everything done, release the video capture object
    cap.release()
    result_vid.release()