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
test_video2 = 'test_video2.mp4'

dataset = "default"

yrange=(400,680)

ds.augmentNonVehicleDatasource(target_size=20000)

#need to find a viable method to deal with hummungus dataset (> 30gb)
#ds.recordingsToDatasource_CrowdAI('object-detection-crowdai')
#ds.recordingsToDatasource_Autti('object-dataset')

ds.datasourceToDataset(dataset)

if not os.path.exists(classifiers_path + classifier_name):
    cc = CarClassifier(yrange=yrange)

    scaler = ds.computeFeaturesForDataset(dataset, cc)
    cc.scaler = scaler

    dataset = ds.loadDataset(dataset)
    print(type(dataset))
    for label in dataset:
        print(label)

    cc.trainSVC(dataset['features'], dataset['y'])
    cc.save(classifiers_path + classifier_name)
else:
    cc = CarClassifier.load(classifiers_path + classifier_name)


trk = tk.Tracker()

window_max = yrange[1]-yrange[0]
window_min = 64
windows_steps = 4
threshold = 4


live = False
display = None

def parseFrame(img):
    global display

    bboxes=cc.findCars(img, windows_range=(window_min,window_max), windows_steps=windows_steps, threshold=threshold, debug=not live)
    detection_list = trk.processFrameDetections(bboxes)
    detection_boxes = []
    for detection in detection_list:
        detection_boxes.append(detection.averagedBox())

    img = utils.draw_boxes(img, bboxes, color=(255, 0, 0))
    img = utils.draw_boxes(img, detection_boxes, color=(0,255,0))

    if live:
        plt.ion()
        if display is None:
            display = plt.imshow(img)
        else:
            display.set_data(img)
        plt.pause(0.05)
    else:
        utils.showImage(img)



    return img


#cc.processVideo(test_video2,parseFrame)
cc.processVideo(test_video,parseFrame)
#cc.processVideo(src_video,parseFrame)





