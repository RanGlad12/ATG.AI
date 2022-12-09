import torch
import os
#from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

def barbell_tracker_train(save_best=False):
    print('Start training yolov5')
    os.chdir('yolov5')
    os.system('python train.py --img 416 --batch 16 --epochs 150 --data ../Gym_barbell-1/data.yaml --weights yolov5s.pt --cache --cfg ../custom_yolov5s.yaml')
    if save_best:
        os.system('cp /content/yolov5/runs/train/exp/weights/best.pt ..')
    os.chdir('..')
    

def barbell_tracker_detect(video_path):
    os.chdir('yolov5')
    os.system('python detect.py --weights /content/drive/MyDrive/best.pt --img 416 --conf 0.4 --source {video_path} --save-txt')
    os.chdir('..')