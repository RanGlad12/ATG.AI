import torch
import os
from IPython.display import Image, clear_output  # to display images

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Import barbell dataset from Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="t6tj7mQYwRntDdaHocGQ")
project = rf.workspace("ecnu").project("gym_barbell")
dataset = project.version(1).download("yolov5")