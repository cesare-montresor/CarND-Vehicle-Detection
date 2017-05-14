from shapely.geometry import box, Polygon
import numpy as np

class Tracker():
    def __init__(self):
        self.current_frame = 0
        self.detections = []
        self.detections_purged = []
        self.min_detection_valid = 15
        self.min_detection_purge = 10
        self.min_area_matching = 0.6  # 70%
        self.min_area_deduplicate = 0.8  # 90%
        self.opposite_lane_polygon = Polygon(((0, 600), (600, 400), (600, 0), (0, 0)))
        self.very_close_polygon = Polygon(((300, 550), (1100, 550), (1100, 720), (300, 720)))

    # assume that is called once per frame, only
    def processFrameDetections(self, bboxes):
        matches = []
        new_detections = []
        for bbox in bboxes:
            position = bboxToPosition(bbox)
            if not self.isValidPosition(position):
                continue
            match = self.findByPosition(position)

            if match is None:
                match = Detection()
                new_detections.append(match)

            match.detected(self.current_frame, position)

        self.current_frame += 1
        self.detections.extend(new_detections)
        self.purge()

        # should i mark a car even if i can't see it this one frame based on the principle that car don't dematerialize ?
        for detection in self.detections:
            if detection.counter >= self.min_detection_valid:
                matches.append(detection)

        to_hide=[]
        # remove double detections
        for detection1 in matches:
            for detection2 in matches:
                if detection1 != detection2:
                    pos1 = detection1.averagedPosition()
                    pos2 = detection2.averagedPosition()
                    if pos1.contains(pos2):
                        detection1.detected(self.current_frame, position)
                        detection2.detected(self.current_frame, position)
                        # print('purge because contained')
                        to_hide.append(detection2)
                    else:
                        if pos1.intersection(
                                pos2).area > pos2.area * self.min_area_deduplicate:  # if it overlaps by xx% of the smaller one
                            # print('purge because overlaps')
                            detection1.detected(self.current_frame, position)
                            detection2.detected(self.current_frame, position)
                            to_hide.append(detection2)

        for detection in to_hide:
            if detection in matches:
                matches.remove(detection)

        return matches

    def isValidPosition(self,position): #filter by shape, size taking into account prospective
        ratio_factor = 3
        ratio_limit = (1/ratio_factor,ratio_factor) # 1/4
        min_size = 20
        max_size = 400
        min_size_close_objects = 100

        xmin,ymin,xmax,ymax = position.bounds
        center = position.centroid
        area = position.area
        w,h = xmax-xmin, ymax-ymin

        # filter opposite lane
        if self.opposite_lane_polygon.contains(center):
            print("opposite lane", w, h)
            return False

        # is right in front of the car
        if self.very_close_polygon.contains(center):
            if w < min_size_close_objects or h < min_size_close_objects:
                print("too small for being so close", w, h)
                return False

        if w < min_size or h < min_size: #too small
            print("too small, size", w,h)
            return False

        if w > max_size or h > max_size: #too big
            print("too big, size", w, h)
            return False

        ratio = w/h
        if not ratio_limit[0] < ratio < ratio_limit[1]: #wrong shape
            print("drop wrong shape", ratio)
            return False





        return True

    def purge(self):
        to_purge = []
        for detection in self.detections:
            if self.current_frame - detection.last_apperence > self.min_detection_purge:
                to_purge.append(detection)


        for detection in to_purge:
            if detection in self.detections:
                self.detections.remove(detection)
                self.detections_purged.append(detection)

    def findByPosition(self, position):
        bestMatch = None
        bestMatch_area=0
        for i, detection in enumerate(self.detections):
            last_position = detection.averagedPosition()
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
        self.position_history = []  # insert from top, because i like it better this way
        self.position_history_frame = []  # insert from top, because i like it better this way
        self.best_position = None


    def detected(self,frame_num, position):
        if self.first_apperence is None:
            self.first_apperence = frame_num

        if self.last_apperence is None or self.last_apperence < frame_num:
            self.counter += 1
            self.last_apperence = frame_num

        self.position_history.insert(0, position)
        self.position_history_frame.insert(0, frame_num)
        self.position_history[0] = self.averagedPosition()



    def averagedPosition(self):
        bbox = self.averagedBox()
        return bboxToPosition(bbox)


    def averagedBox(self, lastN=10):
        avg_bounds = None
        lastN_positions = self.position_history[:lastN]
        for position in reversed(lastN_positions):
            xmin,ymin,xmax,ymax = position.bounds
            bonds = np.array([xmin,ymin,xmax,ymax])
            if avg_bounds is None:
                avg_bounds = bonds
            else:
                avg_bounds += bonds
                avg_bounds = avg_bounds / 2
        avg_bounds = avg_bounds.tolist()
        return ((int(avg_bounds[0]),int(avg_bounds[1])),(int(avg_bounds[2]),int(avg_bounds[3])))


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
