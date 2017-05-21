import numpy as np
import utils
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
import cv2

video_output_path = './videos/'
window_size = (64,64)

def predictionToHeatmap(img, prediction_map, threshold=0.8, debug=False):
    heatmap = np.zeros(img.shape[:2], np.float)
    h,w = img.shape[:2]
    yscale = heatmap.shape[0] / prediction_map.shape[0]
    xscale = heatmap.shape[1] / prediction_map.shape[1]

    if debug: utils.showImage(prediction_map, cmap='hot')
    prediction_map[prediction_map < threshold] = 0
    if debug: utils.showImage(prediction_map, cmap='hot')
    heatmap_regions, cnt_regions = label(prediction_map)

    for region_number in range(1, cnt_regions + 1):
        nonzero = (heatmap_regions == region_number).nonzero()
        for x,y in zip(nonzero[1],nonzero[0]):
            value = prediction_map[y][x]
            #print(x, y, value)

            sx = x * xscale
            sy = y * yscale
            wx = window_size[1] / 2
            wy = window_size[0] / 2
            minx, miny, maxx, maxy = int(sx-wx), int(sy-wy), int(sx+wx), int(sy+wy)

            minx = np.clip(minx, 0, w)
            miny = np.clip(miny, 0, h)
            maxx = np.clip(maxx, 0, w)
            maxy = np.clip(maxy, 0, h)

            #print(miny,maxy,minx,maxx)
            heatmap[miny:maxy, minx:maxx] += 1 #value


    if debug: utils.showImage(heatmap, cmap='hot')
    return heatmap

def heatmapToBoxes(heatmap_img, yoffset, threshold=1, debug=False):
    heatmap_img = heatmap_img.astype(np.int32)  # ensure that the heatmap is integers or label goes a bit funny
    print('heatmap range:', np.min(heatmap_img), np.max(heatmap_img))
    #heatmap_img[heatmap_img<=1] = 0 # remove noise
    bboxes = []
    heatmap_regions, cnt_regions = label(heatmap_img)
    if debug:
        utils.showImage(heatmap_regions)
        print(cnt_regions)

    for region_number in range(1,cnt_regions+1):
        nonzero = (heatmap_regions == region_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)+yoffset), (np.max(nonzerox), np.max(nonzeroy)+yoffset))
        bboxes.append(bbox)

    return bboxes, heatmap_img
