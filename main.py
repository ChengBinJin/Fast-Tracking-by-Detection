import os
# import sys
import cv2
import numpy as np
import tensorflow as tf

from model import Model
# from cropper import CropBatcher
from tracker import Tracker
from drawer import Drawer, RecordVideo


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('data', 'video_01.avi', 'test video path, default: video_01.avi')
tf.flags.DEFINE_float('resize_ratio', 0.5, 'resize ratio of the input frame, default: 0.5')
tf.flags.DEFINE_bool('is_tracker', True, 'use tracker or not, default: True')
tf.flags.DEFINE_bool('is_set_roi', True, 'set roi of the image to give SSD, default: True')
tf.flags.DEFINE_bool('is_draw_roi', True, 'set roi area, default: True')
tf.flags.DEFINE_bool('is_record', False, 'recroding video, default: True')
tf.flags.DEFINE_integer('delay', 1, 'interval between two frames')


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index

    # cropper = CropBatcher(FLAGS)
    model = Model(FLAGS)  # Initialize detection network
    tracker = Tracker(max_age=15, min_hits=3)  # Initialize tracker
    drawer = Drawer(FLAGS)  # Inidialize drawer class
    video_writer = RecordVideo(FLAGS.is_record, FLAGS)

    cv2.namedWindow(FLAGS.data)
    cv2.moveWindow(FLAGS.data, 2500, 300)  # Set display window position to easy show
    cap = cv2.VideoCapture(os.path.join('../data', FLAGS.data))
    frameID, total_time = 0, 0.
    dets_arr, batch_imgs = None, np.asarray([])
    while cap.isOpened():
        ret, raw_frame = cap.read()
        if ret is False:
            print(' [!] Can not read video frame!')
            break

        tic = cv2.getTickCount()  # tic
        # if frameID != 0:
        #     batch_imgs = cropper.feed(raw_frame, dets_arr)

        dets_arr, labels_arr = model.test(raw_frame, batch_imgs)  # run network only in this line

        if FLAGS.is_tracker:  # tracking
            dets_arr = tracker.update(dets_arr)

        toc = cv2.getTickCount()  # toc
        fps = cv2.getTickFrequency() / (toc - tic)  # fps
        total_time += (toc - tic)

        show_frame = drawer.draw(raw_frame, dets_arr, labels_arr, fps)
        # cropper.imshow()
        cv2.imshow(FLAGS.data, show_frame)
        if FLAGS.is_record is True:  # record video
            video_writer.output.write(show_frame)

        # if cv2.waitKey(FLAGS.delay) & 0xFF == 27:
        #     sys.exit(' [*] Esc clicked!')

        frameID += 1
        print('frameID: {}'.format(frameID))

    print('Avg. FPS: {}'.format(1. / (total_time / cv2.getTickFrequency() / frameID)))


if __name__ == '__main__':
    tf.app.run()
