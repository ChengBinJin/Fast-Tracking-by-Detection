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
 
## Detector & Tracker
- Detector uses Inception-SSD ([Single-Shot Multibox Detector](https://arxiv.org/pdf/1512.02325.pdf)) from the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). You can replace the detector by your own detector. Good detector means more powerful tracking-by-detection results.  

- Tracker uses Kalman filter with hungarian data-association algorithm.

## Requirements
- tensorflow 1.10.0
- opencv 3.3.1
- numpy 1.15.2
- filterpy 1.4.5
- sklearn 0.20.0

## Tracking-by-Detection Demo
Three videos are set by different intervals (1, 2, and 3) of the detection. Big interval value gives faster processing time but low accuracy. The hyper-parameter should trade off between speed and accuracy.   
- [Click to go to the full demo on YouTube! Tracking-by-Detection interval 1](https://www.youtube.com/watch?v=HpKMLA19zkg&feature=youtu.be)  
- [Click to go to the full demo on YouTube! Tracking-by-detection interval 2](https://www.youtube.com/watch?v=HpKMLA19zkg&feature=youtu.be)  
- [Click to go to the full demo on YouTube! Tracking-by-detection nterval 3](https://www.youtube.com/watch?v=HpKMLA19zkg&feature=youtu.be)

<p align = 'center'>
  <a href = 'https://www.youtube.com/watch?v=HpKMLA19zkg&feature=youtu.be'>
    <img src = 'https://user-images.githubusercontent.com/37034031/42082312-9ffdbffc-7bc2-11e8-9dfe-e505e5b3c528.gif'>
  </a>
</p>
