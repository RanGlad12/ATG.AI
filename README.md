# ATG.AI - the automatic squat depth judge
<p align="center">
  <img src="https://user-images.githubusercontent.com/112420956/206929746-cfbc65f9-4143-4bba-8328-e21092bcd21d.gif" alt="animated" width="240" height="320" />
</p>

ATG.AI or ass-to-grass.AI is an automatic computer vision squat depth judge based on deep learning. In powerlifting and CrossFit competitions a squat is only valid if the hip crease moves below the height of the top of the knees. However, detecting whether a squat was deep enough or not is a challenging problem. In powerlifting competitions each squat is observed by three judges: one from each side of the lifter and one from the front. Two out of the three judges must agree that the squat was deep enough for the lift to be valid. In CrossFit, a single judge scores the lifter, but some events may include tens or even hundreds of squats. The goal of ATG.AI,  is to eventually assist or replace judges in such competitions, and to help atheletes training for such competitions in assessing their squat depth. 

ATG.AI consists of two machine learning steps. First, a barbell image dataset was used to train a YoloV5 object detector with a single class. The trained YoloV5 is then used to track the barbell in a video of a person doing a squat or squats. The barbell movement in the image y axis is extracted and the peaks corresponding to the deepest point of the squat are marked. A few frames in a window around each peak are then fed into an EfficientNet classifier network trained on images of deep and shallow squats. If a certain percentage of the images are classified as a deep squat we denote the squat as a valid squat, and otherwise it is an invalid squat.

Eventually the project might evolve to include additional movements with different standards such as the clean and jerk, snatch, deadlift, bench press etc. using a similar approach. Another potential goal is to port it to a mobile app so that users can get a more immediate feedback instead of analyzing videos offline.

## Current limitations
Validation accuracy of the classifier is 87% on single images and I have not yet produced a proper annotated video test dataset. It is best to use videos taken from a perspective or side view. Both the barbell and squat datasets contain a low number of images captured from the front and even less so directly from the back of the lifter. Therefore both the tracker and classifier's accuracy is degraded for those angles. I assume the video contains a barbell with plates loaded and that the movements performed are a set of either back squats or front squats captured by a stationary camera. In addition the code operates under the assumption of a single barbell and a single person in the video, though it should work well with a few sporadic detections (or mis-detections) of additional barbells. Spotters assisting with the squat can currently confuse the classifier as they are usually performing a shallow squat to assist the lifter.

## Installation
There are two options for how to install this project.
### In Google Colab
Clone the repository to get ATG_AI.ipynb (you may alternatively just copy its text and create your own .ipynb), open https://colab.research.google.com/ in your browser and load the ATG_AI.ipynb notebook. Click run all. You may upload your own video to Colab and enter its path in the argument for the last cell: 
```
!python squat_judge.py path/of/your/video
```
Once the notebook has finished running simply download your video to watch the results. 
### IDE
Clone the repository, update the yolov5 submodule repository by running  
```
git submodule init
git submodule update
```
in the terminal, then run
```
pip install -r requirements.txt
pip install -r yolov5/requirements.txt
```
## Usage
Run squat_judge.py. A pop-up window will open asking to pick a video file to analyze. Choose a video file, click submit and close the window. After the program is done running a video file result_video.avi will be generated. Supported formats are all the formats supported by YoloV5, though I have only tested on .mp4, .mov and .avi. By default training for the tracker and classifier networks is disabled. You may enable them in suqat_judge.py by setting
```
TRAIN_TRACKER = True
TRAIN_CLASSIFIER = True
```
Note that the dataset for the classifier is private and is not included in this repository - check the Credits section for details regarding how to obtain it.

Parameters to consider changing are the prominence of the detected peaks, width of the detected peaks, and the size of the window of frames considered in the classification.

## Credits
This work could not be done without the terrific dataset created by Paritosh Parmar, Amol Gharat, and Helge Rhodin from the University of British Columbia and FlexAI.Inc. Please check Paritosh's GitHub for their project: https://github.com/ParitoshParmar/Fitness-AQA and if you find their work useful please cite it. You may also fill a request form to get their dataset, of which I used to shallow squat error dataset.

```
@article{parmar2022domain,
  title={Domain Knowledge-Informed Self-Supervised Representations for Workout Form Assessment},
  author={Parmar, Paritosh and Gharat, Amol and Rhodin, Helge},
  journal={arXiv preprint arXiv:2202.14019},
  year={2022}
}
```
