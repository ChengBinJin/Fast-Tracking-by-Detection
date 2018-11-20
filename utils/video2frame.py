import sys
import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data', dest='data', default='video_00.avi', help='video path')
parser.add_argument('--delay', dest='delay', type=int, default=1, help='interval between two frames')
args = parser.parse_args()


def main(dataFolder):
    print('Video path: {}'.format(os.path.join(dataFolder, args.data)))
    cv2.namedWindow('Show')
    cv2.moveWindow("Show", 2500, 100)

    imgSavePath = os.path.join(dataFolder, 'imgs')
    if not os.path.isdir(imgSavePath):
        os.makedirs(imgSavePath)

    videoPath = os.path.join(dataFolder, args.data)
    cap = cv2.VideoCapture(videoPath)
    frameID = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            return

        if frameID == 0:
            print('Frame shape: {}'.format(frame.shape))

        cv2.imshow('Show', frame)  # show frame
        if cv2.waitKey(args.delay) & 0XFF == 27:
            sys.exit('Esc clicked!')

        # save frame
        cv2.imwrite(os.path.join(imgSavePath, str(frameID).zfill(4) + '.jpg'), frame)
        print('Frame ID: {}'.format(frameID))  # pring frame ID
        frameID += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    data_folder = '../../data'
    main(data_folder)
