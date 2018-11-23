import os
import cv2
import numpy as np


class Drawer(object):
    def __init__(self, flags):
        self.flags = flags

        self.redColor = (0, 51, 255)
        self.whiteColor = (255, 255, 255)
        self.thickness = 2
        self.fontSize = 0.7
        self.fontFace = cv2.FONT_HERSHEY_TRIPLEX
        self.max_id = 1000

        self._init_params()

    def _init_params(self):
        self.labels = ["Back", "Car", "Ped", "Cycl", "Motor", "Truck", "Bus", "Van"]

    def draw(self, img, dets, labels, fps):
        h, w = img.shape[0:2]
        show_img = img.copy()

        dets[dets < 0.] = 0.  # some kalman predictions results are negative
        dets = dets.astype(np.uint16)

        for idx in range(dets.shape[0]):
            cv2.rectangle(show_img, (dets[idx, 1], dets[idx, 0]), (dets[idx, 3], dets[idx, 2]),
                          self.redColor, self.thickness)

            # Add label
            if self.flags.is_tracker is False:
                label_id = labels[idx]
                label_str = self.labels[label_id]
                label_size = cv2.getTextSize(label_str, self.fontFace, self.fontSize, self.thickness)
                bottom_left = (dets[idx, 1], dets[idx, 0] + int(0.5 * label_size[0][1]))
                cv2.rectangle(show_img, (dets[idx, 1], dets[idx, 0] - int(0.5 * label_size[0][1])),
                              (dets[idx, 1] + label_size[0][0], dets[idx, 0] + int(0.7 * label_size[0][1])),
                              self.redColor, thickness=-1)
                cv2.putText(show_img, label_str, bottom_left, self.fontFace, self.fontSize, self.whiteColor,
                            thickness=1)

            # Add object ID
            if self.flags.is_tracker:
                id_str = str(np.mod(dets[idx, 4], self.max_id))
                id_size = cv2.getTextSize(id_str, self.fontFace, self.fontSize, self.thickness)
                bottom_left = (dets[idx, 3] - id_size[0][0], dets[idx, 2])
                cv2.rectangle(show_img, (dets[idx, 3] - id_size[0][0], dets[idx, 2] - id_size[0][1]),
                              (dets[idx, 3], dets[idx, 2]), self.redColor, thickness=-1)
                cv2.putText(show_img, id_str, bottom_left, self.fontFace, self.fontSize, self.whiteColor, thickness=1)

        # Draw processing time
        FPS_str = "FPS: {:.1f}".format(fps)
        inform_size = cv2.getTextSize(FPS_str, self.fontFace, self.fontSize, self.thickness)
        margin_h, margin_w = int(0.01 * h), int(0.01 * w)
        bottom_left = (margin_w, inform_size[0][1] + margin_h)
        cv2.putText(show_img, FPS_str, bottom_left, self.fontFace, self.fontSize, self.redColor, self.thickness)

        # Draw ROI region
        if self.flags.is_draw_roi:
            cv2.line(show_img, (0, 240), (1280, 240), self.redColor, self.thickness)

        return show_img


class RecordVideo(object):
    def __init__(self, is_record, flags, height=720, width=1280):
        self.is_record = False
        self.flags = flags
        self.folder = '../videos'

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        if is_record:
            self.video_name = os.path.join(self.folder, '{}_is_track_{}_resize_ratio_{}.avi'.format(
                self.flags.data[:-4], self.flags.is_tracker, self.flags.resize_ratio))
            # define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.output = cv2.VideoWriter(self.video_name, fourcc, 30.0, (width, height))
            self.is_record = True

    def turn_off(self):
        if self.is_record is True:
            self.output.release()
