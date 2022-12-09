import re
import numpy as np
import os

def find_tracker_peaks(frames, labels_dir):
    x = []
    y = []
    ind = [(m.start(0), m.end(0)) for m in re.finditer(r"_\d*.txt", frames[0])]
    ind_end = [(m.start(0), m.end(0)) for m in re.finditer(r"_\d*.txt", frames[-1])]

    start_frame = int(frames[0][ind[0][0] + 1:ind[0][1] - 4])
    end_frame =   int(frames[-1][ind_end[0][0] + 1:ind_end[0][1] - 4])
    output_frame_nums = [int(frame[ind[0][0] + 1:ind[0][1] - 4]) for frame in frames]
    all_frames = np.arange(0, end_frame + 1, dtype=int)

    i = 0
    for frame_num in all_frames:
        frame = frames[i]
        ind = [(m.start(0), m.end(0)) for m in re.finditer(r"_\d*.txt", frames[i])]
        if frame_num != int(frame[ind[0][0] + 1:ind[0][1] - 4]):
            try:
                x.append(x_center)
                y.append(y_center)
            except:
                x.append(0.3)
                y.append(0.3)
        else:
            with open(os.path.join(labels_dir, frame)) as f:
                lines = f.readlines()
                line = lines[0] #TODO: deal with multiple detections per frame
                line = line.split() 
                line = [float(item) for item in line]
                x_center = (line[1] + line[3]) / 2
                y_center = (line[2] + line[4]) / 2
                x.append(x_center)
                y.append(y_center)

                i += 1
    return x, y