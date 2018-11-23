import cv2
import numpy as np


class CropBatcher(object):
    def __init__(self, flags):
        self.flags = flags
        self.size = 128
        self.batch_imgs = np.asarray([])
        self.coordinates = np.asarray([])
        self.num_dets = 0

    def feed(self, img, boxes, scale=0.5):
        boxes[boxes < 0.] = 0.
        boxes = boxes.astype(np.uint16)

        batch_imgs = []
        for idx in range(boxes.shape[0]):
            new_box = boxes[idx, :]
            h, w = new_box[2] - new_box[0], new_box[3] - new_box[1]

            if h == 0 or w == 0:
                continue
            else:
                new_box[0] = np.maximum(0, (new_box[0] - int(h * scale / 2)))
                new_box[1] = np.maximum(0, (new_box[1] - int(w * scale / 2)))
                new_box[2] = np.minimum(img.shape[0], (new_box[2] + int(h * scale / 2)))
                new_box[3] = np.minimum(img.shape[1], (new_box[3] + int(w * scale / 2)))

                cropped_img = img[new_box[0]:new_box[2], new_box[1]:new_box[3]]  # make img bigger and crop
                factor = np.minimum(self.size / cropped_img.shape[0], self.size / cropped_img.shape[1])
                resized_img = cv2.resize(
                    cropped_img, (int(cropped_img.shape[1] * factor), int(cropped_img.shape[0] * factor)),
                    interpolation=cv2.INTER_LINEAR)

                fixed_size_img = np.zeros((self.size, self.size, resized_img.shape[2]))
                # margin h and w
                h_, w_ = int((self.size - resized_img.shape[0]) / 2), int((self.size - resized_img.shape[1]) / 2)
                fixed_size_img[0+h_:resized_img.shape[0]+h_, 0+w_:resized_img.shape[1]+w_] = resized_img
                # fixed_size_img = cv2.resize(cropped_img, (128, 128))
                batch_imgs.append(fixed_size_img)

        self.batch_imgs = np.asarray(batch_imgs)
        self.num_dets = self.batch_imgs.shape[0]

        return self.batch_imgs

    def imshow(self):
        if self.num_dets > 0:
            window_name = "Corp Img"
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 1950, 100)

            if np.mod(self.batch_imgs.shape[0], 2) != 0:
                n_cols = int(self.batch_imgs.shape[0] / 2) + 1
                n_rows = int((self.batch_imgs.shape[0] + 1) / n_cols)
            else:
                n_cols = int(self.batch_imgs.shape[0] / 2)
                n_rows = int(self.batch_imgs.shape[0] / n_cols)

            canvas = np.zeros((self.size * n_rows, self.size * n_cols, self.batch_imgs.shape[3]), dtype=np.uint8)
            #  print('canvas shape: {}'.format(canvas.shape))
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    img_idx = row_idx * n_cols + col_idx
                    if img_idx < self.batch_imgs.shape[0]:
                        canvas[row_idx*self.size:(row_idx+1)*self.size, col_idx*self.size:(col_idx+1)*self.size, :] = \
                            self.batch_imgs[img_idx]

            cv2.imshow(window_name, canvas)
