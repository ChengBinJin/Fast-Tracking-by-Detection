import os
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('gpu_index', '0', 'gpu index, default: 0')
tf.flags.DEFINE_string('video', 'video')

if __name__ == '__main__':
    print('Hello object detection!')
