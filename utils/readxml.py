import os
import sys
import cv2
import argparse
from xml.etree.ElementTree import parse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', dest='data', default='../../data/imgs', help='img and xml folder')
parser.add_argument('--delay_interval', dest='delay_interval', type=int, default=0, help='interval between two frames')
parser.add_argument('--resize_ratio', dest='resize_ratio', type=float, default=1., help='resize ratio of the img')
args = parser.parse_args()


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for  fname in os.listdir(path)]
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


def read_xml(imgFolder, resize_ratio=0.5, delay_interval=1, is_show_frame=False, color=(0, 51, 255), thickness=2):
    img_paths = all_files_under(imgFolder, extension='.jpg')

    if is_show_frame:
        window_name = 'Show'
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 2500, 100)

        for idx, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            img = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
            labels, boxes = read_data(img_path, resize_ratio=resize_ratio)  # load GT
            print('img shape: {}'.format(img.shape))

            # draw bounding box
            if labels is not None:
                for sub_idx, label in enumerate(labels):
                    print(label, boxes[sub_idx])
                    cv2.rectangle(img, (boxes[sub_idx][0], boxes[sub_idx][1]),
                                  (boxes[sub_idx][2], boxes[sub_idx][3]), color, thickness)

            cv2.imshow(window_name, img)
            if cv2.waitKey(delay_interval) & 0XFF == 27:
                sys.exit(' [*] Esc clicked!')

        cv2.destroyAllWindows()


if __name__ == '__main__':
    read_xml(imgFolder=args.data, resize_ratio=args.resize_ratio, delay_interval=args.delay_interval,
             is_show_frame=True)
