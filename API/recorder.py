import os
import cv2
import numpy as np
from datetime import datetime


class RecordVideo(object):
    def __init__(self, FLAGS, img_shape, resize_ratio=1.0, folder='./videos'):
        self.flags = FLAGS
        self.img_shape = img_shape
        self.resize_ratio = resize_ratio
        self.folder = folder

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        cur_time = datetime.now().strftime("%Y%m%d-%H%M")
        self.video_name = os.path.join(self.folder, '{}.avi'.format(cur_time))
        # define the codec and craete VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output = cv2.VideoWriter(
            self.video_name, fourcc, 30.0, (
                int(2 * self.resize_ratio * self.img_shape[1]), int(self.img_shape[0] * self.resize_ratio)))

    def record(self, upper_frame_, bottom_frame_):
        upper_frame = cv2.resize(upper_frame_, None, fx=self.resize_ratio, fy=self.resize_ratio,
                                 interpolation=cv2.INTER_CUBIC)
        bottom_frame = cv2.resize(bottom_frame_, None, fx=self.resize_ratio, fy=self.resize_ratio,
                                  interpolation=cv2.INTER_CUBIC)

        canvas = np.hstack((upper_frame, bottom_frame))
        self.output.write(canvas)

    def turn_off(self):
        self.output.release()
