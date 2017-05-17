import numpy as np
import utils
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

video_output_path = './videos/'


def processVideo(path, processFrame, live=False, debug=False):
    strdate = '_' + utils.standardDatetime()
    output_video = video_output_path + utils.filenameAppend(path, strdate)
    video = VideoFileClip(path)
    video_clip = video.fl_image(processFrame)
    video_clip.write_videofile(output_video, audio=False)


def extractBoxes(heatmap, yoffset=0, threshold=0.7, debug=False):
    if debug: utils.showImage(heatmap, cmap='hot')
    heatmap[heatmap < threshold] = 0
    heatmap[heatmap != 0] = 1
    if debug: utils.showImage(heatmap, cmap='hot')

    final_bboxes = heatmapToBoxes(heatmap, yoffset, debug=debug)
    return final_bboxes

def heatmapToBoxes(heatmap_img, yoffset, debug=False):
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

    return bboxes
