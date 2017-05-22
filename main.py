import dataset as ds
import utils
import os
import matplotlib.pyplot as plt
import tracker as tk
import model as md
from keras.models import load_model
from moviepy.editor import VideoFileClip
import numpy as np
from scipy.ndimage.measurements import label
import classifier as cls
import cv2

classifiers_path = './classifiers/'
video_output_path = './videos/'
classifier_name = 'default.p'
src_video = 'project_video.mp4'
test_video = 'test_video.mp4'

train = False
train_weights = None # './models/cls_cnn_20170518-220630_05-0.01172-0.98876.h5'
predict_weights = "./models/cls_cnn_20170519-120820_03-0.01811-0.98367.h5"


dataset = "default"

image_size=(1280,720)
yrange=(390,680)

xsize = image_size[0]
ysize = yrange[1] - yrange[0]


#ds.recordingsToDatasource_CrowdAI('object-detection-crowdai')
#ds.recordingsToDatasource_Autti('object-dataset')
#ds.augmentNonVehicleDatasource()  # target_size=None -> balance classes

ds.addFlippedToDatasoruce()
ds.datasourceToDataset(dataset)

if train:
    model_name = md.train(dataset, load_weights=train_weights, debug=False)


model = md.classifierCNN([ysize, xsize, 3], load_weights=predict_weights, debug=True)

live = False
display = None
include_pip = True #picture in picture for prediction and heatmap, make debug so much easier
pip_margin = 20

traker = tk.Tracker()

def parseFrame(img):
    global display, yrange
    valid_boxes = []
    img_tosearch = img[yrange[0]:yrange[1], :, :]

    prediction = model.predict(img_tosearch[None, :, :, :], batch_size=1)
    prediction_map = prediction[0, :, :, 0]  # extract 2D prediction map
    prediction_map = cv2.GaussianBlur(prediction_map, (3, 3), 0) #smooth the area of prediction, much cheaper than resizing the image with bicubic
    # utils.showImage(prediction_map,cmap='hot')

    heatmap = cls.predictionToHeatmap(img_tosearch, prediction_map, threshold=0.7, debug=False)

    bboxes, heatmap = cls.heatmapToBoxes(heatmap, yoffset=yrange[0], debug=False) # return cleanup version
    detections = traker.processFrameDetections(img_tosearch, bboxes)
    for detection in detections:
        valid_boxes.append(detection.averagedBox())

    img = utils.draw_boxes(img, bboxes, color=(255, 0, 0))
    img = utils.draw_boxes(img, valid_boxes, color=(0,255,0))


    if include_pip:
        img_h,img_w = img.shape[:2]
        pip_h,pip_w = prediction_map.shape[:2]
        pip_h, pip_w = pip_h*4,pip_w*4
        prediction_pip = cv2.resize(cv2.cvtColor(np.uint8(prediction_map*255), cv2.COLOR_GRAY2RGB), (pip_w,pip_h) )
        heatmap_pip = cv2.resize(cv2.cvtColor(np.uint8((heatmap/np.max(heatmap))*255), cv2.COLOR_GRAY2RGB), (pip_w,pip_h) )

        img[pip_margin:pip_h + pip_margin, pip_margin:pip_w + pip_margin, :] = prediction_pip
        img[pip_margin:pip_h + pip_margin, img_w-(pip_margin+pip_w):img_w-pip_margin, :] = heatmap_pip

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


def processVideo(path, processFrame, live=False, debug=False):
    strdate = '_' + utils.standardDatetime()
    output_video = video_output_path + utils.filenameAppend(path, strdate)
    video = VideoFileClip(path)
    video_clip = video.fl_image(processFrame)
    video_clip.write_videofile(output_video, audio=False)


processVideo(src_video,parseFrame)





