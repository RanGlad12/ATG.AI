# ATG.AI

ATG.AI is an automatic computer vision squat depth judge based on deep learning. In powerlifting and CrossFit competitions a squat is only valid if the hip crease moves below the height of the top of the knees. However, detecting whether a squat was deep enough or not is a challenging problem. In powerlifting competitions each squat is observed by three judges: one from each side of the lifter and one from the front. Two out of the three judges must agree that the squat was deep enough for the lift to be valid. In CrossFit, a single judge scores the lifter, but some events may include tens or even hundreds of squats. The goal of ATG.AI, or ass-to-grass.AI is to eventually assist or replace judges in such competitions, and to help atheletes training for such competitions in assessing their squat depth. 

ATG.AI consists of two machine learning steps. First, a barbell image dataset was used to train a YoloV5 object detector with a single class. The trained YoloV5 is then used to track the barbell in a video of a person doing a squat or squats. The barbell movement in the image y axis is extracted and the peaks corresponding to the deepest point of the squat are marked. A few frames in a window around the peak are then fed into an EfficientNet classifier network trained on images of deep and shallow squats. If a certain percentage of the images are classified as a deep squat we denote the squat as a valid squat, and otherwise it is an invalid squat.

Eventually the project might evolve to include additional movements with different standards such as the clean and jerk, snatch, deadlift, bench press etc. using a similar approach. 

## Installation
Clone the repository, and run pip install -r requirements.txt in the terminal.

## Usage
Run squat_judge.py. A pop-up window will open asking to pick a video file to analyze. Supported formats are all the formats supported by YoloV5, though I have only tested on .mp4 and .avi. 

## Credits
This work could not be done without the terrific dataset created by Paritosh Parmar, Amol Gharat, and Helge Rhodin from the University of British Columbia and FlexAI.Inc. Please check Paritosh's GitHub for their project: https://github.com/ParitoshParmar/Fitness-AQA

```
@article{parmar2022domain,
  title={Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment},
  author={Parmar, Paritosh and Gharat, Amol and Rhodin, Helge},
  journal={arXiv preprint arXiv:2202.14019},
  year={2022}
}
```
