import cv2
import numpy as np


class Drawer(object):
    def __init__(self, flags, img_shape=(720, 1280)):
        self.flags = flags
        self.img_shape = img_shape

        self.redColor = (0, 51, 255)
        self.whiteColor = (255, 255, 255)
        self.thickness = 2
        self.fontSize = 0.7
        self.fontFace = cv2.FONT_HERSHEY_TRIPLEX
        self.max_id = 1000

        self._init_params()

    def _init_params(self):
        self.labels = ["Back", "Car", "Ped", "Cycl", "Motor", "Truck", "Bus", "Van"]

    @staticmethod
    def cal_iou(bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        iou = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (
                bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return iou

    def hide_boxes(self, dets):
        # delete some boxes that included in the big box
        # x1, y1, x2, y2, id, is_dect:[0, 1]
        num_objects = dets.shape[0]
        flags = np.ones(num_objects, dtype=np.uint8)

        new_dets = []
        for idx_a in range(num_objects):
            # print('ID: {}, Shape: {}'.format(dets[idx_a][4], dets[idx_a][:4]))

            if (dets[idx_a][2] - dets[idx_a][0] < 0.05 * self.img_shape[0]) or (
                    dets[idx_a][3] - dets[idx_a][1] < 0.05 * self.img_shape[1]):
                flags[idx_a] = 0
                # print(' [*] Delete ID: {}'.format(dets[idx_a][4]))
                continue

            for idx_b in range(idx_a+1, num_objects):
                if flags[idx_b] == 0:
                    continue

                iou = self.cal_iou(dets[idx_a], dets[idx_b])

                if iou >= 0.3:
                    if dets[idx_a, 5] == 0: # (false, false) and (false, true)
                        flags[idx_a] = 0
                        break
                    else:
                        if dets[idx_b, 5] == 0: # (true, false)
                            flags[idx_b] = 0
                            continue
                        else: # (true, true)
                            flags[idx_a] = 0
                            break

        for idx in range(num_objects):
            if flags[idx] == 1:
                new_dets.append(dets[idx])

        return np.asarray(new_dets)


    def draw(self, img, dets, labels, fps):
        h, w = img.shape[0:2]
        show_img = img.copy()

        dets[dets < 0.] = 0.  # some kalman predictions results are negative
        dets = dets.astype(np.uint32)

        # Hide some uncorrect boxes
        dets = self.hide_boxes(dets)

        for idx in range(dets.shape[0]):
            cv2.rectangle(show_img, (dets[idx, 1], dets[idx, 0]), (dets[idx, 3], dets[idx, 2]),
                          self.redColor, self.thickness)

            # # Add label
            # label_id = labels[idx]
            # label_str = self.labels[label_id]
            # label_size = cv2.getTextSize(label_str, self.fontFace, self.fontSize, self.thickness)
            # bottom_left = (dets[idx, 1], dets[idx, 0] + int(0.5 * label_size[0][1]))
            # cv2.rectangle(show_img, (dets[idx, 1], dets[idx, 0] - int(0.5 * label_size[0][1])),
            #               (dets[idx, 1] + label_size[0][0], dets[idx, 0] + int(0.7 * label_size[0][1])),
            #               self.redColor, thickness=-1)
            # cv2.putText(show_img, label_str, bottom_left, self.fontFace, self.fontSize, self.whiteColor,
            #             thickness=1)

            # Add object ID
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

        return show_img
