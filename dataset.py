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

trainfile = 'train.npz'
validfile = 'valid.npz'
testfile = 'test.npz'

datasources_path = './datasources/'
recordings_path = './recordings/'
datasets_path = './datasets/'
video_output_path = './videos/'
default_batch_size = 32


def loadDatasetGenerators(name, batch_size=default_batch_size):  # return generator
    basepath = datasets_path + name + '/'
    train_dataset = loadDataset(basepath + trainfile)
    valid_dataset = loadDataset(basepath + validfile)

    train_size = len(train_dataset['path'])
    valid_size = len(valid_dataset['path'])
    # print(train_dataset.keys())

    sample = utils.loadImage(train_dataset['path'][0])
    sample_shape = sample.shape
    sample_type = type(sample[0][0][0])

    info = {
        'n_train':train_size,
        'n_train_batch': math.ceil(train_size/batch_size),
        'n_valid':valid_size,
        'n_valid_batch': math.ceil(valid_size/batch_size),
        'input_shape': sample_shape,
        'data_type': sample_type
    }

    return datasetGenerator(train_dataset, batch_size, augment=False), datasetGenerator(valid_dataset, batch_size), info

def datasetGenerator(dataset, batch_size=default_batch_size, augment=False):
    n_dataset = len(dataset)
    paths = dataset['path']
    labels = dataset['y']

    while 1:
        paths, labels = shuffle(paths, labels)
        for offset in range(0, n_dataset, batch_size):
            batch_paths = paths[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]
            X = []
            y = []
            for path,label in zip(batch_paths, batch_labels):
                img = utils.loadImage(path)
                if augment:
                    img = augmentImage(img)

                X.append(img)
                y.append([[[label]]])  # output convolution layer is of shape (?,1,1,1), matching same shape

            X = np.array(X)
            y = np.array(y)
            yield shuffle(X,y)



def datasourceToDataset(name=None, valid_split=0.2, test_split=0.2, reindex_only=False, force=False):
    if name is None: name = 'dataset_'+utils.standardDatetime()
    basepath = datasets_path + name + '/'
    basepath_img = basepath + 'img/'

    if os.path.exists(basepath+trainfile) and not force and not reindex_only:
        return basepath


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

    train_paths, test_paths,  train_labels, test_labels  = train_test_split(paths,       labels,       test_size=test_split)
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size=valid_split)

    train_dataset = {'path': train_paths, 'y': train_labels}
    valid_dataset = {'path': valid_paths, 'y': valid_labels}
    test_dataset  = {'path': test_paths,  'y': test_labels}

    saveDataset(basepath + trainfile, train_dataset)
    saveDataset(basepath + validfile, valid_dataset)
    saveDataset(basepath + testfile,  test_dataset)

    return basepath

def addFlippedToDatasoruce(force=False):
    src_vehicles_path = datasources_path + 'vehicles/*/*.png'
    src_nonvehicles_path = datasources_path + 'non-vehicles/*/*.png'

    dst_vehicles_path = datasources_path + 'vehicles/flipped/'
    dst_nonvehicles_path = datasources_path + 'non-vehicles/flipped/'
    if not force and os.path.exists(dst_vehicles_path) and os.path.exists(dst_nonvehicles_path):
        return

    utils.makedirs(dst_nonvehicles_path)
    utils.makedirs(dst_vehicles_path)

    dst_vehicles_paths = glob.glob(dst_vehicles_path + '*.png')
    dst_nonvehicles_paths = glob.glob(dst_nonvehicles_path + '*.png')

    cleanup_paths = np.concatenate((dst_vehicles_paths,dst_nonvehicles_paths))
    [os.remove(path) for path in cleanup_paths]

    src_vehicles_paths = glob.glob(src_vehicles_path)
    src_nonvehicles_paths = glob.glob(src_nonvehicles_path)

    for i,img_path in enumerate(src_vehicles_paths):
        filename = utils.filename(img_path)
        filename = utils.filenameAppend(filename,'_'+str(i))
        print(img_path,filename)
        img = utils.loadImage(img_path)
        img = cv2.flip(img,1)
        utils.saveImage(dst_vehicles_path + filename, img)

    for i,img_path in enumerate(src_nonvehicles_paths):
        filename = utils.filename(img_path)
        filename = utils.filenameAppend(filename,'_'+str(i))
        img = utils.loadImage(img_path)
        img = cv2.flip(img,1)
        utils.saveImage(dst_nonvehicles_path + filename, img)


def augmentNonVehicleDatasource(target_size=None, force=False):
    vehicles_path = datasources_path + 'vehicles/*/*.png'
    src_nonvehicles_path = datasources_path + 'non-vehicles/*/*.png'
    dst_nonvehicles_path = datasources_path + 'non-vehicles/augmented/'

    utils.makedirs(dst_nonvehicles_path)

    vehicles_img_paths = glob.glob(vehicles_path)
    nonvehicles_img_paths = glob.glob(src_nonvehicles_path)
    augmented_img_paths = glob.glob(dst_nonvehicles_path)


    if force:
        for path in augmented_img_paths:
            os.remove(path)
        augmented_img_paths = []

    if target_size is None:
        target_size = len(vehicles_img_paths)-(len(nonvehicles_img_paths) + len(augmented_img_paths))
    print('target size:', target_size)

    if target_size <= 0:
        print("Augmenting non-vehicle datasource:", "[SKIP]")
        return 0


    last_progress = 0
    items_cnt = 0

    src_paths = shuffle(glob.glob(src_nonvehicles_path))
    aug_per_img = int(math.ceil(target_size / len(src_paths)))
    for img_path in src_paths:
        if items_cnt > target_size:
            break
        filename = utils.filename(img_path)
        img = utils.loadImage(img_path)
        for i in range(aug_per_img):
            aug_img = augmentImage(img)
            aug_img_path = dst_nonvehicles_path + utils.filenameAppend(filename,'_'+str(i))
            utils.saveImage(aug_img_path,aug_img, cspace=cv2.COLOR_RGB2BGR)

            items_cnt += 1
            progress = int((items_cnt * 100) / target_size)
            if progress > last_progress:
                print('Augmenting non-vehicle datasource:', progress, '%')
                last_progress = progress
            if items_cnt > target_size:
                break






def recordingsToDatasource_CrowdAI(name, size=(64, 64), force=False):
    filters = {
        5: 'Car'
    }
    return recordingsToDatasource(name, key_frame=4, key_topleft=(0, 1), key_bottomright=(2, 3), key_filter=filters, size=size, force=force)

def recordingsToDatasource_Autti(name, size=(64, 64), force=False):
    filters = {
        6: 'car',
        5: '0'
    }
    return recordingsToDatasource(name, key_frame=0, key_topleft=(1,2),key_bottomright=(3,4), key_filter=filters, size=size, csvdelimiter=' ', force=force)


def recordingsToDatasource(name, key_frame, key_topleft, key_bottomright, key_filter={}, size=(64,64), csvdelimiter=',', force=False):
    src_basepath = recordings_path + name + '/'
    dst_basepath = datasources_path + 'vehicles/' + name + '/'

    utils.makedirs(dst_basepath)

    if len(glob.glob(dst_basepath+'*.png')) > 0 and not force:
        return dst_basepath,0

    size_ratio = size[0]/size[1]

    csvpath = src_basepath + csvname
    lines = utils.loadCSV(csvpath,delimiter=csvdelimiter)

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
    items_cnt = 0
    for frame in byframe_list:
        bboxes = byframe_list[frame]
        img_path = src_basepath + frame
        img = utils.loadImage(img_path)
        for i, bbox in enumerate(bboxes):
            framename = utils.filenameAppend(frame, "_" + str(i))
            framename = utils.replaceExtension(framename,'png')
            framepath = dst_basepath + framename

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


            img_section = img[h_min:h_max,w_min:w_max,:]
            img_section = cv2.resize(img_section,size, interpolation=cv2.INTER_CUBIC)
            utils.saveImage(framepath, img_section, cspace=cv2.COLOR_BGR2RGB)
            items_cnt +=1
            frame_num += 1
            progress = int((frame_num * 100) / cnt)
            if progress > last_progress:
                print('Generating datasources for ', name, progress, '%')
                last_progress = progress
    return dst_basepath, items_cnt


def computeFeaturesForDataset(name, classifier, debug=False, force=False):
    dataset = loadDataset(name)

    for label in dataset:
        print(label)

    scaler = None
    if 'features' not in dataset or force:
        features = classifier.extractFeatures(dataset['path'], debug=debug)
        features = np.array(features).astype(np.float64)
        scaler = StandardScaler().fit(features)
        scaled_features = scaler.transform(features)
        dataset['features'] = scaled_features
        saveDataset(name, dataset)

    return scaler



def saveDataset(path, data):
    np.savez(path, **data)


def loadDataset(path):
    dataset = None
    with np.load(path) as npzfile:
        dataset = {}
        for key in npzfile.keys():
            dataset[key] = npzfile[key].tolist()
    return dataset

def loadDatasets(names):
    dataset = None
    for name in names:
        dataset_part = loadDataset(name)
        if dataset is None:
            dataset = dataset_part
        else:
            for key in dataset_part:
                dataset[key].extends(dataset_part[key])
    return dataset

# video parsing and extraction

def processVideo(self, path, function, live=False, debug=False):
    if not live:
        strdate = '_' + utils.standardDatetime()
        output_video = video_output_path + utils.filenameAppend(path, strdate)
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

def augmentImage(img):
    action_list = [randomHue, randomBrightness, randomNothing, randomShadows] # randomRotation
    action_num = np.random.randint(0, len(action_list))
    action = action_list[action_num]
    aug_img = action(img)
    return aug_img

def randomBrightness(img, limit=0.4):
    img_new = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img_new = np.array(img_new, dtype = np.float64)
    img_new[:,:,2] = img_new[:,:,2] * (np.random.uniform(low=limit, high=2-limit))
    img_new[:,:,2][img_new[:,:,2]>255] = 255 #cap values
    img_new = np.array(img_new, dtype = np.uint8)
    img_new = cv2.cvtColor(img_new,cv2.COLOR_HSV2RGB)
    return img_new

def randomHue(img):
    hue = np.random.randint(low=0, high=255)
    img_new = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img_new = np.array(img_new, dtype = np.float64)
    img_new[:,:,0] = (img_new[:,:,0] + hue ) % 255
    img_new = np.array(img_new, dtype = np.uint8)
    img_new = cv2.cvtColor(img_new,cv2.COLOR_HSV2RGB)
    return img_new

def randomRotation(img, max_rotation = 30):
    height, width, depth = img.shape
    angle = int( max_rotation*np.random.uniform(-1,1) )
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    img_new = cv2.warpAffine(img, M, (width, height))
    return img_new

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

def randomNothing(img):
    return img
