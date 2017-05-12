import dataset as ds
from classifier import CarClassifier
import utils
import os
import matplotlib.pyplot as plt
import tracker as tk

classifiers_path = './classifiers/'
classifier_name = 'default.p'
src_video = 'project_video.mp4'
test_video = 'test_video.mp4'

dataset = "default"
recording_crowdai = 'object-detection-crowdai'
recording_autti = 'object-dataset'

yrange=(400,680)

# switch CSV file as dataset
#ds.recordingsToDataset_CrowdAI(recording_crowdai)
#ds.computeFeaturesForDataset(recording_crowdai, debug=False, force=True)
#ds.recordingsToDataset_Autti(recording_autti, reindex_only=True)
#ds.computeFeaturesForDataset(recording_autti, debug=False)


a = ((1,2),(3,4))

p = tk.bboxToPosition(a)
b = tk.positionToBbox(p)
print(b)



if not os.path.exists(classifiers_path + classifier_name):
    cc = CarClassifier(yrange=yrange)

    ds.datasourceToDataset(dataset)
    basepath,scaler = ds.computeFeaturesForDataset(dataset, cc, debug=False, force=True)
    cc.scaler = scaler

    dataset = ds.loadDataset(dataset)
    cc.trainSVC(dataset['features'], dataset['y'])
    cc.save(classifiers_path + classifier_name)
else:
    cc = CarClassifier.load(classifiers_path + classifier_name)


trk = tk.Tracker()

window_max = yrange[1]-yrange[0]
window_min = 64
windows_steps = 4


display = None

def parseFrame(img):
    global display
    bboxes=cc.findCars(img, windows_range=(window_min,window_max), windows_steps=4)
    detection_list = trk.processFrameDetections(bboxes)
    detection_boxes = []
    for detection in detection_list:
        detection_boxes.append(detection.getBbox())

    img2 = utils.draw_boxes(img, bboxes, color=(255, 0, 0))
    img2 = utils.draw_boxes(img2,detection_boxes, color=(0,255,0))
    plt.ion()
    if display is None:
        display = plt.imshow(img2)
    else:
        display.set_data(img2)
    plt.pause(0.05)
    return img2



cc.processVideo(test_video,parseFrame)
#cc.processVideo(src_video,parseFrame)



