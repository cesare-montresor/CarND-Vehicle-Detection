import glob, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from shutil import copyfile
import datetime
import pickle
import csv
import pygame


## IMAGE DISPLAY

def showImages(images, cols=None, rows=None, cmap=None):
    if len(images) == 1:
        showImage(images[0],cmap=cmap)
        return

    if rows is None and cols is None:
        rows = cols = int(math.ceil(math.sqrt(len(images))))
    if rows is None:
        rows = int(math.ceil(len(images) / cols))
    if cols is None:
        cols = int(math.ceil(len(images) / rows))

    if type(images[0]) == type(""):
        images = list(map(lambda image_path:cv2.imread(image_path),images))

    i = 0
    f, sub_plts = plt.subplots(rows, cols)
    for r in range(rows):
        for c in range(cols):
            sub_plts[r, c].axis('off')
            if i<len(images):
                sub_plts[r,c].imshow(images[i],cmap=cmap)
                i += 1

    plt.show()
    plt.close('all')

def showImage(image, cmap=None):
    if type(image) == type(""):
        image = cv2.imread(image)
    plt.imshow(image, cmap=cmap)
    plt.show()
    plt.close()

def drawGrid(img,rows=10,cols=10):
    img = img.copy()
    h,w,d = img.shape
    dh = h / rows
    dw = w / cols
    for r in range(rows):
        for c in range(cols):
            cv2.line(img, (0, int(dh*r)), (w,int(dh*r)), (255, 0, 0), 5) # horizontal
            cv2.line(img, ( int(dw*c), 0), ( int(dw*c), h), (0, 255, 0), 5) # vertical
    return img


## IMAGE MODIFICATION

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def cropImage(image,margins): # css style: top, right, bottom, left
    h,w,d = image.shape
    return image[margins[1]:w-margins[3], margins[0]:h-margins[2]]

def color_space(image, cspace=None):
    if cspace == 'HSV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCb':
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        return image.copy()

def normalizeImage(img):
    img = img.copy()
    if np.max(img) <= 1:  # convert bitmask into image
        img *= 255

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

## IMAGE LOAD

def saveImage(path, img, cspace = None):
    if cspace is not None:
        img = cv2.cvtColor(img, cspace)
    cv2.imwrite(path,img)
    pass

def loadImage(path, cspace = cv2.COLOR_BGR2RGB):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cspace)
    return img

def loadImages(path, cspace = cv2.COLOR_BGR2RGB):
    img_paths = glob.glob(path)
    imgs = []
    for img_path in img_paths:
        img = loadImage(img_path,cspace=cspace)
        imgs.append(img)
    return imgs

## FS OPERATIONS

def replaceExtension(path, ext):
    parts = path.split('.')
    parts[-1] = ext
    return ".".join(parts)


def filenameAppend(path, suffix):
    parts = path.split(".")
    ext = parts[-1]
    base = ".".join(parts[:-1])+suffix+'.'+ext
    return base

def filename(path):
    parts = path.split('/')
    if len(parts) > 0:
        return parts[-1]
    else:
        return path

def copy(src,dst):
    if not os.path.isfile(src):
        return None
    parts = dst.split('/')
    os.makedirs("/".join(parts[:-1]), exist_ok=True)
    return copyfile(src, dst)

def loadData(path):
    if not os.path.exists(path):
        return None

    value=None
    ext = path.split('.')[-1]
    if ext == 'jpg':
        value = cv2.imread(path)
    elif ext == 'p':
        with open(path, 'rb') as pfile:
            value = pickle.load(pfile)
    return value

def saveData(path,data):
    print('saveData path', path)
    print('saveData type', type(data))
    ext = path.split('.')[-1]
    if ext == 'jpg':
        cv2.imwrite(path, data)
    elif ext == 'p':
        with open(path, 'wb') as pfile:
            pickle.dump(data, pfile)
    return True

def loadCSV(path, delimiter=',', quotes='"'):
    if not os.path.exists(path):
        return None
    lines = []
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=delimiter, quotechar=quotes)
        lines = list(csvreader)
    return lines

def makedirs(path):
    os.makedirs(path, exist_ok=True)

## other

def standardDatetime():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


## play sounds

def play(path):
    sound = pygame.mixer.Sound(path)
    sound.play()
