import os
import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, flags):
        if self._init_graph():  # read graph and initialize pb file
            print(' [*] Read graph SUCCESS!')

        self.flags = flags
        self.ori_h, self.ori_w = 0, 0
        self.margin_h = 0.

        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.detection_graph, config=run_config)

    def _init_graph(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(os.path.join('../models', 'frozen_inference_graph.pb'), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Gets handles to input and output tensors
        ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}

        self.output_tensor = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:  # output
                self.output_tensor[key] = self.detection_graph.get_tensor_by_name(tensor_name)

        self.input_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

    def set_roi(self, img):
        self.margin_h = int(img.shape[0] / 3)
        roi_img = img[self.margin_h::, :, :]
        self.ori_h, self.ori_w = roi_img.shape[0:2]

        return roi_img

    def test(self, img):
        input_img = np.expand_dims(self.set_roi(img), axis=0)
        output_dict = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: input_img})

        # All outputs are float32 numpy arrays, so convert types as appropriate
        num_dets = int(output_dict['num_detections'][0])
        full_dets = output_dict['detection_boxes'][0]
        labels = output_dict['detection_classes'][0].astype(np.uint8)[:num_dets]

        dets = []
        for idx in range(num_dets):  # from [0., 1.] to original width and height range
            det = [self.margin_h + full_dets[idx][0] * self.ori_h, full_dets[idx][1] * self.ori_w,
                   self.margin_h + full_dets[idx][2] * self.ori_h, full_dets[idx][3] * self.ori_w]

            dets.append(det)

        return np.asarray(dets), np.asarray(labels)
