from __future__ import print_function

import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter


def iou_tracker(bb_test, bb_gt):
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


def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2
    y = bbox[1] + h / 2
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox, min_hits):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(  # state transistion matrix
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(  # measurement function
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.  # measurement uncertainty / noise
        self.kf.P[4:, 4:] *= 1000.  # covariance matrix
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01  # process uncertainty / noise
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)  # filter state estimate
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.vip = False
        self.min_hits = min_hits
        self.previous = self.kf.x
        self.skip_frame = 0
        self.obj_speed = 0
        self.max_age_ = 0
        self.obj_delete = False

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.previous = self.kf.x
        self.obj_delete = False
        self.max_age_, self.obj_speed = self.maximum_age()

        if self.hit_streak >= self.min_hits:
            self.vip = True

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.previous = self.kf.x
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        self.obj_delete = True
        if self.time_since_update == 0:
            self.max_age_ = 0
            self.obj_speed = 0

        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

    def maximum_age(self):
        obj_speed = np.sqrt(self.kf.x[4] ** 2 + self.kf.x[5] ** 2) * 10
        if obj_speed < 10:
            self.skip_frame = 100 / (obj_speed + 0.01)
            # self.skip_frame = 10 * np.exp(-obj_speed)
        elif obj_speed < 100:
            self.skip_frame = 150 / (obj_speed + 0.01)
        else:
            self.skip_frame = 10

        if self.skip_frame < 3:
            self.skip_frame = 3
        if self.skip_frame > 20:
            self.skip_frame = 20

        return int(self.skip_frame), obj_speed


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou_tracker(trk, det)

    # Solve the linear assignment problem using the Hungarian algorithm
    # The problem is also known as maximum weight matching in bipartite graphs. The method is also known as the
    # Munkres or Kuhn-Munkres algorithm.
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)  # store index

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)  # store index

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Tracker(object):
    def __init__(self, max_age=15, min_hits=1, det_confidence=0.5):
        self.max_age = max_age  # 保持多少帧，我们允许多少帧没有检测到 mis_detection frame
        self.min_hits = min_hits  # 检测多少帧后就稳定了要开始做tracking，即消除false detection
        self.trackers = []
        self.frame_count = 0
        self.det_confidence = det_confidence

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))

        to_del = []
        ret1 = []
        for t, trk in enumerate(trks):  # t: index, trk: content
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmateched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[
                    np.where(matched[:, 1] == t)[0], 0]  # np.where returns two array, one is row index, another
                # is column. [[]]->[] dimension changed from (a,b) to (a)
                trk.update(dets[d, :][0])

        for i in unmateched_dets:
            trk = KalmanBoxTracker(dets[i, :], min_hits=self.min_hits)
            self.trackers.append(trk)

        i = len(self.trackers)
        # print("============================================")
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]  # trk.get_state() is [[]], shape is (1,4), trk.get_state()[0] is [] (shape: (4,))
            # [[1,2]] is a 2d array, shape is (1,2), every row has two elements
            # [1,2] is a one-d array, shape is (2,), this array has two elements
            # [[1],[2]] is a 2d array, shape is (2,1) every row has one elements
            self.max_age = trk.max_age_

            if trk.obj_delete and np.abs(trk.previous[0] - d[0] > 70):
                trk.time_since_update = 20

            for trk1 in self.trackers:
                d1 = trk1.get_state()[0]
                over_lap = iou_tracker(d, d1)
                if trk1.obj_delete is False and trk.obj_delete and ((d[0] > d1[0] and d[1] > d1[1] and d[2] + d[0] < d1[
                    2] + d1[0] and d[3] + d[1] < d1[3] + d1[1]) or (
                        d[0] > d1[0] and d[1] > d1[1] and d[2] < d1[2] and d[3] < d1[3])):
                    trk.time_since_update = 20

                if trk1.obj_delete is False and trk.obj_delete and (over_lap >= 0.3 and over_lap <= 0.99):
                    trk.time_since_update = 20

            if (trk.time_since_update < self.max_age) and ((trk.hit_streak >= self.min_hits) or trk.vip):
                ret1.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret1) > 0:
            return np.concatenate(ret1)
        else:
            return np.empty((0, 5))
