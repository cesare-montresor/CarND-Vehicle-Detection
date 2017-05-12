from shapely.geometry import box
import numpy as np

class Tracker():
    def __init__(self):
        self.current_frame = 0
        self.detections = []
        self.detections_purged = []
        self.min_detection_valid = 5
        self.min_detection_purge = 5
        self.min_area_matching = 0.7 #70%

    # assume that is called once per frame, only
    def processFrameDetections(self, bboxes):
        matches = []
        new_detections = []
        for bbox in bboxes:
            position = bboxToPosition(bbox)
            match = self.findByPosition(position)

            if match is None:
                match = Detection()
                new_detections.append(match)

            match.detected(self.current_frame, position)

            if match.counter >= self.min_detection_valid:
                matches.append(match)

        self.current_frame += 1
        self.detections.extend(new_detections)
        self.purge()

        return matches

    def purge(self):
        purge_idxs = []
        for i,detection in enumerate(self.detections):
            if self.current_frame - detection.last_apperence > self.min_detection_purge:
                purge_idxs.append(i)

        for idx in purge_idxs:
            purges_detection = self.detections.pop(idx)
            self.detections_purged.append(purges_detection)


    def findByPosition(self, position):
        bestMatch = None
        bestMatch_area=0
        for i, detection in enumerate(self.detections):
            last_position = detection.position_history[0]
            last_area = last_position.area
            match = position.intersection(last_position)
            match_area = match.area

            if match_area > last_area*self.min_area_matching and match_area > bestMatch_area:
                bestMatch = detection
                bestMatch_area = match_area

        return bestMatch

    pass

class Detection():
    autoincrement = 0

    @staticmethod
    def getID():
        id = Detection.autoincrement
        Detection.autoincrement += 1
        return id

    def __init__(self):
        self.id = Detection.getID()
        self.counter = 0
        self.first_apperence = None #frame num
        self.last_apperence = None # frame num
        self.position_history = [] #insert from top


    def detected(self,frame_num, position):
        self.counter += 1
        if self.first_apperence is None:
            self.first_apperence = frame_num
        self.last_apperence = frame_num
        self.position_history.insert(0, position)

    def getBbox(self,history=0):
        position = self.position_history[history]
        return positionToBbox(position)


def bboxToPosition(bbox):
    # box(minx, miny, maxx, maxy, ccw=True):
    return box(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])

def positionToBbox(position):
    # [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    minx, miny, maxx, maxy = position.bounds
    bbox = ((int(minx),int(miny)), (int(maxx),int(maxy)))
    return bbox
