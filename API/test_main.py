import os
import sys
import cv2
import tensorflow as tf

from model import Model
from tracker import Tracker
from drawer import Drawer

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('data', '../../data/video_01.avi', 'test video path, default: ../data/video_01.avi')
tf.flags.DEFINE_integer('delay', 1, 'interval between two frames, default: 1')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    img_shape = (720, 1280, 2)
    model = Model(FLAGS)  # Initialize detection network
    drawer = Drawer(FLAGS, img_shape)  # Inidialize drawer class
    tracker = Tracker(img_shape, min_hits=3)  # Initialize tracker

    window_name = 'Tracking-by-Detection'
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 0, 0)  # set display window position to easy showing
    cap = cv2.VideoCapture(FLAGS.data)
    frameID = 0
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if ret is False:
            print(' [!] Can not read vidoe frame!')
            break

        tic = cv2.getTickCount()  # tic
        dets_arr, labels_arr = model.test(raw_frame)
        dets_arr = tracker.update(dets_arr)
        # tracker
        toc = cv2.getTickCount()  # toc
        fps = cv2.getTickFrequency() / (toc - tic)  # fps

        show_frame = drawer.draw(raw_frame, dets_arr, labels_arr, fps)

        cv2.imshow(window_name, show_frame)
        if cv2.waitKey(FLAGS.delay) & 0xFF == 27:
            sys.exit(' [*] Esc clicked!')

        frameID += 1
        print('frameID: {}'.format(frameID))


if __name__ == '__main__':
    tf.app.run()
