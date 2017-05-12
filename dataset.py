import glob, math
import numpy as np
import h5py, pickle
import datetime
import utils
from sklearn.utils import shuffle
import cv2
import os
import features as ft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
import csv

csvname = 'labels.csv'
datasetfile = 'dataset.p'
datasources_path = './datasources/'
recordings_path = './recordings/'
datasets_path = './datasets/'
video_output_path = './videos/'
default_batch_size = 32

def recordingsToDataset_CrowdAI(name, size=(64, 64), reindex_only=False, force=False):
    filters = {
        5: 'Car'
    }
    return recordingsToDataset(name, key_frame=4, key_topleft=(0, 1), key_bottomright=(2, 3), key_filter=filters, size=size, force=force, reindex_only=reindex_only)

def recordingsToDataset_Autti(name, size=(64, 64), reindex_only=False, force=False):
    filters = {
        6: 'car',
        5: '0'
    }
    return recordingsToDataset(name, key_frame=0, key_topleft=(1,2),key_bottomright=(3,4), key_filter=filters, size=size, csvdelimiter=' ', force=force, reindex_only=reindex_only)


def recordingsToDataset(name, key_frame, key_topleft, key_bottomright, key_filter={}, size=(64,64), csvdelimiter=',', reindex_only=False, force=False):
    src_basepath = recordings_path + name + '/'
    dst_basepath = datasets_path + name + '/'
    dst_basepath_img = dst_basepath + 'img/'

    utils.makedirs(dst_basepath_img)

    if os.path.exists(dst_basepath+datasetfile) and not force and not reindex_only:
        return dst_basepath,0

    size_ratio = size[0]/size[1]

    csvpath = src_basepath + csvname
    lines = utils.loadCSV(csvpath,delimiter=csvdelimiter)

    #get only cars with no occlusion (enough?)

    for k in key_filter:
        value = key_filter[k]
        lines = list(filter(lambda x: x[k]==value,lines))
    cnt = len(lines)

    byframe_list = {}
    for line in lines:
        frame, x1,y1,x2,y2 = line[key_frame], line[key_topleft[0]], line[key_topleft[1]], line[key_bottomright[0]], line[key_bottomright[1]]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bbox = [[x1, y1], [x2, y2]]
        if frame not in byframe_list:
            byframe_list[frame] = []
        byframe_list[frame].append(bbox)

    last_progress = 0
    frame_num = 0
    paths = []
    labels = []
    for frame in byframe_list:
        bboxes = byframe_list[frame]
        img_path = src_basepath + frame
        img = utils.loadImage(img_path)
        for i, bbox in enumerate(bboxes):
            framename = utils.filenameAppend(frame, "_" + str(i))
            framepath = dst_basepath_img + framename

            tl,br = bbox
            w,h = abs(tl[1]-br[1]), abs(tl[0]-br[0])
            new_w,new_h = 0,0
            if w/h > size_ratio:
                new_w = int(h / size_ratio)
                new_h = h
            else:
                new_w = w
                new_h = int(w * size_ratio)

            if new_w <= 0 or new_h <=0:
                continue #sometimes is too small and the resize crash :(

            w_min = int((tl[0] + br[0]) / 2 - (new_w / 2))
            w_max = int((tl[0] + br[0]) / 2 + (new_w / 2))
            h_min = int((tl[1] + br[1]) / 2 - (new_h / 2))
            h_max = int((tl[1] + br[1]) / 2 + (new_h / 2))

            if not reindex_only:
                img_section = img[h_min:h_max,w_min:w_max,:]
                img_section = cv2.resize(img_section,size, interpolation=cv2.INTER_CUBIC)
                utils.saveImage(framepath, img_section, cspace=cv2.COLOR_BGR2RGB)

            paths.append(framepath)
            labels.append(1)

            frame_num += 1
            progress = int((frame_num * 100) / cnt)
            if progress > last_progress:
                print('Generating dataset', progress, '%')
                last_progress = progress

    dataset = {'path': paths, 'y': labels}
    utils.saveData(dst_basepath + datasetfile, dataset)




def datasourceToDataset(name=None, limit=None, reindex_only=False, force=False):
    if name is None: name = 'dataset_'+utils.standardDatetime()
    basepath = datasets_path + name + '/'
    basepath_img = basepath+ 'img/'

    if os.path.exists(basepath+datasetfile) and not force and not reindex_only:
        return basepath,0


    vehicles_search_path = datasources_path + 'vehicles/*/*.png'
    nonvehicles_search_path = datasources_path + 'non-vehicles/*/*.png'
    print(vehicles_search_path,nonvehicles_search_path)

    vehicles_paths = glob.glob(vehicles_search_path)
    nonvehicles_paths = glob.glob(nonvehicles_search_path)
    print('vehicles_paths', len(vehicles_paths))
    print('nonvehicles_paths', len(nonvehicles_paths))

    paths = []
    labels = []
    for img_path in vehicles_paths:
        parts = img_path.split('/')
        filename = "_".join(parts[-2:])
        fullpath = basepath_img + filename
        if not reindex_only:
            utils.copy(img_path, fullpath)
        paths.append(fullpath)
        labels.append(1)

    for img_path in nonvehicles_paths:
        parts = img_path.split('/')
        filename = "_".join(parts[-2:])
        fullpath = basepath_img + filename
        if not reindex_only:
            utils.copy(img_path, fullpath)
        paths.append(fullpath)
        labels.append(0)
    dataset_cnt = len(paths)
    dataset = {'path': paths, 'y': labels}

    utils.saveData(basepath + datasetfile, dataset)

    return basepath, dataset_cnt

def computeFeaturesForDataset(name, classifier, debug=False, force=False):
    basepath = datasets_path + name + '/'
    dataset_path = basepath + datasetfile

    dataset = loadData(dataset_path)
    for label in dataset:
        print(label)

    scaler = None
    if 'features' not in dataset or force:
        features = classifier.extractFeatures(dataset['path'], debug=debug)
        features = np.array(features).astype(np.float64)
        scaler = StandardScaler().fit(features)
        scaled_features = scaler.transform(features)
        dataset['features'] = scaled_features
        utils.saveData(basepath + datasetfile, dataset)

    return basepath, scaler


def loadDataset(name):
    dataset = []

    with open(datasets_path + name + '/' + datasetfile, 'rb') as picklefile:
        dataset=pickle.load(picklefile)

    return dataset


def loadData(path):
    value=None
    ext = path.split('.')[-1]
    if ext == 'jpg':
        value = cv2.imread(path)
    elif ext == 'p':
        with open(path, 'rb') as pfile:
            value = pickle.load(pfile)
    return value

# video parsing and extraction

def processVideo(self, path, function, live=False, debug=False):
    if not live:
        strdate = '_' + utils.standardDateFormat()
        output_video = video_output_path + utils.filename_append(path, strdate)
        video = VideoFileClip(path)
        video_clip = video.fl_image(function)
    else:
        vidcap = cv2.VideoCapture(path)
        while True:
            success, image = vidcap.read()
            if not success:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            final_image = self.pipeline(image, debug=debug)
            utils.showImage(final_image)


# augmenting


def randomBrightness(img, limit=0.4):
    img_new = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img_new = np.array(img_new, dtype = np.float64)
    img_new[:,:,2] = img_new[:,:,2] * (np.random.uniform(low=limit, high=2-limit))
    img_new[:,:,2][img_new[:,:,2]>255] = 255 #cap values
    img_new = np.array(img_new, dtype = np.uint8)
    img_new = cv2.cvtColor(img_new,cv2.COLOR_HSV2RGB)
    return img_new

def randomShift(img, steering, max_shift_x = 10, max_shift_y = 10, steering_strenght=1):
    height, width, depth = img.shape
    x_seed = np.random.uniform(-1,1)
    deltaX = max_shift_x * x_seed
    deltaY = max_shift_y * np.random.uniform(-1,1)
    steering += x_seed * steering_strenght
    trans = np.float32([[1, 0, deltaX], [0, 1, deltaY]])
    img_new = cv2.warpAffine(img, trans, (width,height))
    return img_new, steering

def randomShadows(img, max_shadows = 3, min_aplha=0.1, max_aplha=0.8, min_size=0.2, max_size=0.8 ):
    img_new = img.copy()
    height, width, depth = img_new.shape
    # print(width,height)
    shadow_num = int(max_shadows * np.random.uniform())+1
    for i in range(shadow_num):
        x = int(width * np.random.uniform())
        y = int(height * np.random.uniform())
        w2 = int( (width * np.random.uniform(min_size,max_size))/2 )
        h2 = int( (height * np.random.uniform(min_size,max_size))/2 )
        top, bottom = y - h2, y + h2
        left, right = x - w2, x + w2
        top, bottom = max(0, top), min(height, bottom)
        left, right = max(0, left), min(width, right)
        img_new[top:bottom, left:right, :] = img_new[top:bottom, left:right, :] * np.random.uniform(min_aplha,max_aplha)
    return img_new


def histogramEqualizationAndColorSpace(image):
    ycrcb=cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    return ycrcb

def processImage(image):
    image = histogramEqualizationAndColorSpace(image)
    return image

def processImages(images):
    images = [processImage(image) for image in images]
    return images

def standardDateFormat():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def filename_append(path, suffix):
    parts = path.split(".")
    ext = parts[-1]
    base = ".".join(parts[:-1])+suffix+'.'+ext
    return base

def change_ext(path, new_ext):
    parts = path.split(".")
    parts[-1] = new_ext
    return ".".join(parts)




