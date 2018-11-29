import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter

def iou_tracker(bb_test_, bb_gt_):
    bb_test = bb_test_.copy().astype(np.float32)
    bb_gt = bb_gt_.copy().astype(np.float32)

    xx1 = np.maximum(bb_test[0], bb_gt[0])  # x1
    yy1 = np.maximum(bb_test[1], bb_gt[1])  # x2
    xx2 = np.minimum(bb_test[2], bb_gt[2])  # x3
    yy2 = np.minimum(bb_test[3], bb_gt[3])  # x4
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
    return np.array([x, y, s, r]).reshape((4, 1))  # center_x, center_y, area, ratio


def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))  # x1, y1, x2, y2
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    def __init__(self, bbox, min_hits, count=0, num_classes=20, interval=1):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # state transistion matrix
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]])
        # measurement function
        self.kf.H = np.array(
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
        self.id = count
        self.history = []
        self.hits = 0
        self.hit_streak = 1
        self.age = 0
        self.vip = False
        self.min_hits = min_hits

        # add for relative max_age of the object
        self.previous_x = self.kf.x
        self.obj_speed = 0.
        self.max_age = 0.
        self.is_detect = 0
        # add for saving labels
        self.num_classes = num_classes  # 20 means number classes in PASCAL VOC
        self.interval = interval
        self.label_memory = np.zeros((self.num_classes,), dtype=np.uint16)
        self.label_memory[int(bbox[4])] += 1
        self.labelID = np.argmax(self.label_memory)

    def update(self, bbox):
        # bbox: [x1, y1, x2, y2, label_id]
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.previous_x = self.kf.x
        self.is_detect = 1
        self.max_age = self.cal_max_age()
        self.label_memory[int(bbox[4])] += 1
        self.labelID = np.argmax(self.label_memory)

        if self.hit_streak >= self.min_hits:
            self.vip = True

    def predict(self, is_dect=True):
        #
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.previous_x = self.kf.x
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        if is_dect:
            self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        self.is_detect = 0

        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

    def cal_max_age(self):
        # x, y, w, h, dx, dy, ratio(w/h)
        obj_speed = np.sqrt(self.kf.x[4] ** 2 + self.kf.x[5] ** 2)

        if obj_speed < 1:
            skip_frame = np.minimum(20, 10 / (obj_speed + 1e-7))
        elif obj_speed < 10:
            skip_frame = np.maximum(3, 15 / (obj_speed + 1e-7))
        else: # >= 10
            skip_frame = 10

        return int(skip_frame / self.interval)


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
    def __init__(self, img_shape, min_hits=0, num_classes=20, interval=1):
        self.img_shape = img_shape
        self.min_hits = min_hits
        self.num_classes = num_classes
        self.interval = interval
        self.trackers = []
        self.frame_count = 0
        self.kalman_count = 0
        self.skip_ratio = 0.04

    def update(self, obj_dets, obj_labels, is_dect=True):
        self.frame_count += 1

        # delete too small objects
        dets = []
        for idx in range(obj_dets.shape[0]):
            if (obj_dets[idx][2] - obj_dets[idx][0] >= self.skip_ratio * self.img_shape[0]) and (
                    obj_dets[idx][3] - obj_dets[idx][1] >= self.skip_ratio * self.img_shape[1]):
                dets.append(np.hstack((obj_dets[idx], obj_labels[idx])))
        dets = np.asarray(dets)

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret1 = []
        for t, trk in enumerate(trks):  # t: index, trk: content
            pos = self.trackers[t].predict(is_dect=is_dect)[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):  # np.isnan: test element-wise for NaN and return result as a bollean array
                to_del.append(t)

        # np.ma.masked_invalid: Mask an array where invalid values occur (NaNs or infs)
        # np.ma.compress_rows: Suppresss whole rows of a 2-D array that contain masked values
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Hungarian data association
        matched, unmateched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                # np.where returns two array, one is row index, another is column.
                # [[]]->[] dimension changed from (a,b) to (a)
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        for i in unmateched_dets:
            trk = KalmanBoxTracker(dets[i, :], min_hits=self.min_hits, count=self.kalman_count,
                                   num_classes=self.num_classes, interval=self.interval)
            self.kalman_count += 1
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            # trk.get_state() is [[]], shape is (1,4), trk.get_state()[0] is [] (shape: (4,))
            # [[1,2]] is a 2d array, shape is (1,2), every row has two elements
            # [1,2] is a one-d array, shape is (2,), this array has two elements
            # [[1],[2]] is a 2d array, shape is (2,1) every row has one elements
            d = trk.get_state()[0]
            # if (trk.time_since_update < trk.max_age) and ((trk.hit_streak >= self.min_hits) or trk.vip):
            if (trk.time_since_update < trk.max_age) and trk.vip:
                ret1.append(np.concatenate((d, [trk.id + 1], [trk.is_detect], [trk.labelID])).reshape(1, -1))

            i -= 1

            if trk.time_since_update > trk.max_age:
                self.trackers.pop(i)

        if len(ret1) > 0:
            # return np.concatenate(ret1)
            return self.hide_boxes(np.concatenate(ret1), is_dect=is_dect)
        else:
            return np.empty((0, 7))

    @staticmethod
    def hide_boxes(dets, is_dect=True):
        # dets: [x1, y1, x2, y2, objID, is_update(0,1), labelID]
        # delete some boxes that included in the big box
        dets[dets < 0.] = 0.  # some kalman predictions results are negative
        dets = dets.astype(np.uint16)

        # x1, y1, x2, y2, id, is_dect:[0, 1]
        num_objects = dets.shape[0]
        flags = np.ones(num_objects, dtype=np.uint8)

        new_dets = []
        for idx_a in range(num_objects):
            for idx_b in range(idx_a+1, num_objects):
                if flags[idx_b] == 0:
                    continue

                if is_dect:
                    # If A include B, and B is predicted then delete B
                    if (dets[idx_a, 0] <= dets[idx_b, 0]) and (dets[idx_a, 1] <= dets[idx_b, 1]) and (
                        dets[idx_a, 2] >= dets[idx_b, 2]) and (dets[idx_a, 3] >= dets[idx_b, 3]) and (
                            dets[idx_b, 5] == 0):
                            flags[idx_b] = 0
                            continue
                    # B inlcude A, and A is predicted tehn delete A
                    elif (dets[idx_a, 0] >= dets[idx_b, 0]) and (dets[idx_a, 1] >= dets[idx_b, 1]) and (
                        dets[idx_a, 2] <= dets[idx_b, 2]) and (dets[idx_a, 3] <= dets[idx_b, 3]) and (
                            dets[idx_a, 5] == 0):
                            flags[idx_a] = 0
                            break

                iou = iou_tracker(dets[idx_a], dets[idx_b])
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
