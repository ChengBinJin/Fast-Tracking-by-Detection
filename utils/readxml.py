import os
import sys
import cv2
import argparse
import numpy as np
from xml.etree.ElementTree import parse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', dest='data', default='../../data/imgs', help='img and xml folder')
parser.add_argument('--delay', dest='delay', type=int, default=1, help='interval between two frames')
parser.add_argument('--resize_ratio', dest='resize_ratio', type=float, default=0.5, help='resize ratio of the img')
args = parser.parse_args()


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def read_data(filename, resize_ratio=0.5):
    if not os.path.isfile(filename[:-3]+'xml'):
        return None, None
    else:
        labels, boxes = [], []
        node = parse(filename[:-3]+'xml').getroot()

        elems = node.findall('object')
        for subelem in elems:
            # read label
            labels.append(subelem.find('name').text)

            # rad bounding boxes
            box = []
            for idx, item in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
                box.append(int(resize_ratio * int(subelem.find('bndbox').find(item).text)))  # x1, y1, x2, y2
            boxes.append(box)

    return labels, boxes


def crop_imgs(img, boxes, scale=0.5, SIZE=128, delay=1):
    window_name = "Corp Img"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 2500, 800)

    batch_imgs = []
    for idx in range(len(boxes)):
        box = boxes[idx]
        # from (x_min, y_min, x_max, y_max) horizontal x axis to vertical x axis
        new_box = [box[1], box[0], box[3], box[2]]
        h, w = new_box[2] - new_box[0], new_box[3] - new_box[1]
        new_box[0] = np.maximum(0, (new_box[0] - int(h * scale / 2)))
        new_box[1] = np.maximum(0, (new_box[1] - int(w * scale / 2)))
        new_box[2] = np.minimum(img.shape[0], (new_box[2] + int(h * scale / 2)))
        new_box[3] = np.minimum(img.shape[1], (new_box[3] + int(w * scale / 2)))

        cropped_img = img[new_box[0]:new_box[2], new_box[1]:new_box[3]]
        factor = np.minimum(SIZE / cropped_img.shape[0], SIZE / cropped_img.shape[1])
        resized_img = cv2.resize(cropped_img, (int(cropped_img.shape[1] * factor), int(cropped_img.shape[0] * factor)),
                                 interpolation=cv2.INTER_LINEAR)
        fixed_size_img = np.zeros((SIZE, SIZE, resized_img.shape[2]))
        h_, w_ = int((SIZE - resized_img.shape[0]) / 2), int((SIZE - resized_img.shape[1]) / 2)  # margin h and w
        fixed_size_img[0+h_:resized_img.shape[0]+h_, 0+w_:resized_img.shape[1]+w_] = resized_img
        batch_imgs.append(fixed_size_img)

    batch_imgs = np.asarray(batch_imgs)
    # print('batch_imgs shape: {}'.format(batch_imgs.shape))

    if np.mod(batch_imgs.shape[0], 2) != 0:
        n_cols = int(batch_imgs.shape[0] / 2) + 1
        n_rows = int((batch_imgs.shape[0] + 1) / n_cols)
    else:
        n_cols = int(batch_imgs.shape[0] / 2)
        n_rows = int(batch_imgs.shape[0] / n_cols)
    # print('n_rows: {}'.format(n_rows))
    # print('n_cols: {}'.format(n_cols))

    canvas = np.zeros((SIZE * n_rows, SIZE * n_cols, batch_imgs.shape[3]), dtype=np.uint8)
    #  print('canvas shape: {}'.format(canvas.shape))
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            img_idx = row_idx * n_cols + col_idx
            if img_idx < batch_imgs.shape[0]:
                canvas[row_idx * SIZE: (row_idx + 1) * SIZE, col_idx * SIZE: (col_idx + 1) * SIZE, :] = \
                    batch_imgs[img_idx]

    cv2.imshow(window_name, canvas)

    return batch_imgs


def read_xml(imgFolder, resize_ratio=0.5, delay=1, is_show_frame=False, color=(0, 51, 255), thickness=2):
    img_paths = all_files_under(imgFolder, extension='.jpg')

    if is_show_frame:
        window_name = 'Show'
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 2500, 100)

        for idx in range(len(img_paths)):
            img = cv2.imread(img_paths[idx])
            img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
            labels, boxes = read_data(img_paths[idx], resize_ratio=resize_ratio)  # load GT
            print('idx: {}, objects: {}'.format(idx, len(boxes)))
            # print('num objects: {}'.format(len(boxes)))

            batch_imgs = crop_imgs(img, boxes, delay=delay)

            # draw bounding box
            if labels is not None:
                for sub_idx in range(len(labels)):
                    # label = labels[sub_idx]
                    # print(label, boxes[sub_idx])
                    cv2.rectangle(img, (boxes[sub_idx][0], boxes[sub_idx][1]),
                                  (boxes[sub_idx][2], boxes[sub_idx][3]), color, thickness)

            cv2.imshow(window_name, img)
            if cv2.waitKey(delay) & 0XFF == 27:
                sys.exit(' [*] Esc clicked!')

        cv2.destroyAllWindows()


if __name__ == '__main__':
    read_xml(imgFolder=args.data, resize_ratio=args.resize_ratio, delay=args.delay,
             is_show_frame=True)
