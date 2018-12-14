# Fast Tracking-by-Detection
This repository is the fast tracking-by-detection module in ADAS (Advanced Driver Assistance System).   

<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/49984460-89baa080-ffab-11e8-8b5c-50007524d5bd.png" width=800>
</p>

## Pros of the Tracking-by-Detection
- Remove false detections
- Handle unstable detections
- Give object IDs
- Correct category of the object by voting accroding to time series
- Some post-processings are included in tracker (e.g. ignore too small objects and delete some overlapped boxes)  

## Requirements
- tensorflow 1.10.0
- opencv 3.3.1
- numpy 1.15.2
- filterpy 1.4.5
- sklearn 0.20.0

## Tracking-by-Detection Demo
Three videos are set by different intervals (1, 2, and 3) of the detection. Big interval value gives faster processing time but low accuracy. The interval is hyper-parameter, so it should be set by trading off between speed and accuracy.   

- Detection interval == 1.
<p align = 'center'>
  <a href = 'https://www.youtube.com/watch?v=EJkdIyk8JxY'>
    <img src = 'https://user-images.githubusercontent.com/37034031/49987504-d35cb880-ffb6-11e8-9ea2-4c7d5130c84b.gif'>
  </a>
</p>

- Detection interval == 2.
<p align = 'center'>
  <a href = 'https://www.youtube.com/watch?v=e1ig3GEzuJo&t=9s'>
    <img src = 'https://user-images.githubusercontent.com/37034031/49987601-2df61480-ffb7-11e8-9de9-0d43e16a2553.gif'>
  </a>
</p>

- Detection interval == 3.
<p align = 'center'>
  <a href = 'https://www.youtube.com/watch?v=Cinq8BE-eqY&feature=youtu.be'>
    <img src = 'https://user-images.githubusercontent.com/37034031/49987818-0489b880-ffb8-11e8-99bd-c4863f09e5e4.gif'>
  </a>
</p>
**Note:** left is tracking-by-detection, right is detection only.  

- [Click to go to the full demo on YouTube! Tracking-by-Detection interval 1](https://www.youtube.com/watch?v=EJkdIyk8JxY)  
- [Click to go to the full demo on YouTube! Tracking-by-detection interval 2](https://www.youtube.com/watch?v=e1ig3GEzuJo&t=9s)  
- [Click to go to the full demo on YouTube! Tracking-by-detection nterval 3](https://www.youtube.com/watch?v=Cinq8BE-eqY&feature=youtu.be)  

## Implementation Details
- Detector uses Inception-SSD ([Single-Shot Multibox Detector](https://arxiv.org/pdf/1512.02325.pdf)) from the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). You can replace the detector by your own detector. Good detector means more powerful tracking-by-detection results.  
- Tracker uses Kalman filter with hungarian data-association algorithm.  

## Documentation
### Directory Hierarchy
``` 
.
│   Fast-tracking-by-detection
│   ├── data
│   │   ├── video_01.avi (video not included in git)
│   ├── src
│   │   ├── cropper.py
│   │   ├── drawer.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── tracker.py
│   │   │   ├── API
│   │   │   │   ├── drawer.py
│   │   │   │   ├── model.py
│   │   │   │   ├── recorder.py
│   │   │   │   ├── test_main.py
│   │   │   │   └── tracker.py
│   │   │   ├── models
│   │   │   │    └── frozen_inference_graph.pb (Tensorflow compiled pb file)
│   │   │   ├── utils
│   │   │   │   ├── readxml.py
│   │   │   │   └── video2frame.py
```  
**src**: source codes of the fast-tracking-by-detection
**API**: the codes in `API` is the **naive version of the tracking-by-detection**. All of the following demo videos are used naive version. And the advanced version is under progress.

### Test Tracking-by-detection  
Run 'test_main.py' in the `API' folder.  

```
python test_main.py --data=/give/adress/of/the/input/video
```  
- `--gpu_index`: gpu index, default: `0`  
- `--data`: test video path, default: `../data/video_01.avi`  
- `--interval`: detector interval between tracker, default: `2`  
- `--delay`: interval between two frames when showing, default: `1`  
- `--is_record`: recording video, default: `False`  

### Replace Your own Detector
Please refer to the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) tutorial to train your own model and dataset. After training compiled to the `pb` file and replace the `frozen_inference_graph.pb` in the folder `models`.   

You also need to consider about the `labels' of the object. Each model the label indexs are different. In our model, index ordering is "0: Background", "1: Car", "2: Pedestrain", "3: Cyclist", "4: Motorcyclist", "5: Truck", "6: Bus", and "7: Van". You can refer to the file `drawer.py` in the `API` folder. (**Note:** `drawer.py` in `API` and `src` folder is different. `drawer.py` in `src` is for advanced version of the tracking-by-detection)

### Citation
```
  @misc{chengbinjin2018fasttrackingbydetection,
    author = {Cheng-Bin Jin},
    title = {Fast-Tracking-by-Detection},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/Fast-Tracking-by-Detection}},
    note = {commit xxxxxxx}
  }
```  

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
