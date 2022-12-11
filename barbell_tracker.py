import os
import shutil
import torch

# from IPython.display import Image, clear_output  # to display images

print((f"Setup complete. Using torch {torch.__version__}",
       f"({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})"))


def barbell_tracker_train(save_best=False):
    '''
    Trains a yolov5 barbell tracker
    '''
    print('Start training yolov5')
    os.system('python yolov5/train.py --img 416 --batch 16 --epochs 150 --data Gym_barbell-1/data.yaml --weights yolov5s.pt --cache --cfg custom_yolov5s.yaml')
    if save_best:
        shutil.copyfile('yolov5/runs/train/exp/weights/best.pt', '')


def barbell_tracker_detect(video_path):
    '''
    Uses the Yolov5 barbell tracker to detect a barbell in a video.
    Saves the results to yolov5/runs/exp
    '''
    os.system(f'python yolov5/detect.py --weights best.pt --img 416 --conf 0.4 --source {video_path} --save-txt')
