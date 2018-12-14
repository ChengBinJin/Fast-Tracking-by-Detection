import os
import sys
import cv2
import time
import numpy as np
import tensorflow as tf

from model import Model
from tracker import Tracker
from drawer import Drawer
from recorder import RecordVideo

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('data', '../../data/video_01.avi', 'test video path, default: ../data/video_01.avi')
tf.flags.DEFINE_integer('interval', 2, 'detector interval between tracker, default: 2')
tf.flags.DEFINE_integer('delay', 1, 'interval between two frames, default: 1')
tf.flags.DEFINE_bool('is_record', False, 'recording video, default: False')


def main(_):
    if FLAGS.interval > 3:
        sys.exit(' [!] {} is too big detector interval...'.format(FLAGS.interval))

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    img_shape = (720, 1280, 3)
    model = Model(FLAGS)  # Initialize detection network
    tracker = Tracker(img_shape, min_hits=3, num_classes=8, interval=FLAGS.interval)  # Initialize tracker
    drawer = Drawer(FLAGS, img_shape)  # Inidialize drawer class

    recoder = None
    if FLAGS.is_record:
        recoder = RecordVideo(FLAGS, img_shape)

    window_name0 = 'Detection-by-Tracking'
    window_name1 = 'Detection Only'
    cv2.namedWindow(window_name0)
    cv2.namedWindow(window_name1)
    cv2.moveWindow(window_name0, 630, 0)
    cv2.moveWindow(window_name1, 1920, 0)

    cap = cv2.VideoCapture(FLAGS.data)
    frameID = 0
    moving_tra, moving_det = 0., 0.
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if ret is False:
            print(' [!] Can not read vidoe frame!')
            break

        # print('frameID: {}'.format(frameID))

        tic = time.time()
        dets_arr, labels_arr, is_dect = None, None, None
        if np.mod(frameID, FLAGS.interval) == 0:
            dets_arr, labels_arr = model.test(raw_frame)
            is_dect = True
        elif np.mod(frameID, FLAGS.interval) != 0:
            dets_arr, labels_arr = np.array([]), np.array([])
            is_dect = False
        pt_det = (time.time() - tic) # unit is sec.

        tracker_arr = tracker.update(dets_arr, labels_arr, is_dect=is_dect)  # tracker
        pt_tra = (time.time() - tic)

        if frameID != 0:
            moving_tra = (frameID / (frameID + 1) * moving_tra) + (1. / (frameID + 1) * pt_tra)
            moving_det = (frameID / (frameID + 1) * moving_det) + (1. / (frameID + 1) * pt_det)

        show_frame = drawer.draw(raw_frame, tracker_arr, labels_arr, (1. / (moving_tra + 1e-8)), is_tracker=True)
        det_frame = drawer.draw(raw_frame, dets_arr, labels_arr, (1. /(moving_det + 1e-8)), is_tracker=False)

        cv2.imshow(window_name0, show_frame)
        cv2.imshow(window_name1, det_frame)
        if cv2.waitKey(FLAGS.delay) & 0xFF == 27:
            sys.exit(' [*] Esc clicked!')

        frameID += 1

        if FLAGS.is_record:
            recoder.record(show_frame, det_frame)

    if FLAGS.is_record:
        recoder.turn_off()


if __name__ == '__main__':
    tf.app.run()
