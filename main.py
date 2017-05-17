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
test_video2 = 'test_video2.mp4'

train = True
load_weights = None # 'models/cls_cnn_20170516-003449_01-0.04674.h5'


dataset = "default"

image_size=(1280,720)
yrange=(400,680)

xsize = image_size[0]
ysize = yrange[1] - yrange[0]




ds.recordingsToDatasource_CrowdAI('object-detection-crowdai')
ds.recordingsToDatasource_Autti('object-dataset')
ds.augmentNonVehicleDatasource()  # target_size=None -> balance classes
ds.datasourceToDataset(dataset)

if train:
    model_name = md.train(dataset, load_weights=load_weights, debug=False)


model = md.classifierCNN([ysize, xsize, 3], load_weights=load_weights)

live = False

def parseFrame(img):
    global display, yrange
    img_tosearch = img[yrange[0]:yrange[1], :, :]
    prediction = model.predict(img_tosearch[None, :, :, :], batch_size=1)

    print(prediction.shape)
    heatmap = prediction[0,:,:,0]
    print(heatmap.shape)
    min_value, max_value = np.min(heatmap), np.max(heatmap)
    print('Heatmap value range', min_value, max_value)

    heatmap = cv2.resize(heatmap, (img_tosearch.shape[1],img_tosearch.shape[0]) )  #resize for better details?

    bboxes = cls.extractBoxes(heatmap,threshold=0.5, yoffset=yrange[0], debug=True)


    img = utils.draw_boxes(img, bboxes, color=(255, 0, 0))
    #img = utils.draw_boxes(img, detection_boxes, color=(0,255,0))

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


#processVideo(test_video2,parseFrame)
processVideo(test_video,parseFrame)
#processVideo(src_video,parseFrame)





