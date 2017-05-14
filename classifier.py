from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
import numpy as np
import utils
import cv2
from moviepy.editor import VideoFileClip
import features as ft
from scipy.ndimage.measurements import label



video_output_path = './videos/'

class CarClassifier():

    def save(self,path):
        utils.saveData(path, self)

    @staticmethod
    def load(path):
        return utils.loadData(path)


    def __init__(self, cspace='LUV', spatial_size=(16, 16),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                     spatial_feat=True, hist_feat=True, hog_feat=True, yrange=(400,680), window_size=64, debug=False):
        self.color_space = cspace
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.svc = None
        self.scaler = None
        self.yrange = yrange
        self.window_size=window_size
        self.live_canvas = None

    def trainSVC(self, X, y, split=0.2, C=1, benchmark=False):
        print("Training linear SVC")
        X,y = shuffle(X,y)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=split)

        # Train
        svc = LinearSVC(C=C)
        start_time = time.time()
        svc.fit(X_train, y_train)
        train_time = time.time() - start_time
        print('Train time:', int(train_time),'seconds')

        # Accuracy
        accuracy = round(svc.score(X_test, y_test), 4)
        print('Test accuracy:', accuracy)

        # Prediction benchmark
        if benchmark:
            start_time = time.time()
            svc.predict(X_test)
            prediction_time = (time.time() - start_time)/len(X_test)
            prediction_time = round(prediction_time, 4)
            print('Prediction time:', prediction_time,'seconds')
        self.svc = svc
        return svc

    def processVideo(self, path, processFrame, live=False, debug=False):
        strdate = '_' + utils.standardDatetime()
        output_video = video_output_path + utils.filenameAppend(path, strdate)
        video = VideoFileClip(path)
        video_clip = video.fl_image(processFrame)
        video_clip.write_videofile(output_video, audio=False)

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def findCars(self, img, windows_range, windows_steps, threshold=5 ,debug=False):
        ystart,ystop = self.yrange
        scale_range = []
        wsize_range = []
        for i in range(0,windows_steps):
            wsize = ((windows_range[1] - windows_range[0]) / windows_steps) * i + windows_range[0]
            scale = wsize / self.window_size
            scale_range.append(scale)
            wsize_range.append(int(wsize))


        if debug:
            print(wsize_range)
            print(scale_range)

        img = self.normalizeImage(img)
        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = utils.color_space(img_tosearch,self.color_space)
        bboxes = []
        for scale, wsize in zip(scale_range,wsize_range):
            if scale != 1:
                imshape = ctrans_tosearch.shape
                scaled_w,scaled_h = int(np.ceil(imshape[1] / scale)), int(np.ceil(imshape[0] / scale))
                ctrans_tosearch = cv2.resize(ctrans_tosearch, (scaled_w, scaled_h))
            #if debug: utils.showImage(ctrans_tosearch)
            ch1 = ctrans_tosearch[:, :, 0]
            ch2 = ctrans_tosearch[:, :, 1]
            ch3 = ctrans_tosearch[:, :, 2]

            # Define blocks and steps as above
            nxblocks = (ch1.shape[1] // self.pix_per_cell) - self.cell_per_block + 1
            nyblocks = (ch1.shape[0] // self.pix_per_cell) - self.cell_per_block + 1
            nfeat_per_block = self.orient * self.cell_per_block ** 2

            # orginal sampling rate
            window = self.window_size
            nblocks_per_window = (window // self.pix_per_cell) - self.cell_per_block + 1
            cells_per_step = 1  # 75% overlapping
            nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
            nysteps = (nyblocks - nblocks_per_window) // cells_per_step

            # Compute individual channel HOG features for the entire image

            if self.hog_feat:
                hog1 = ft.get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vector=False)
                hog2 = ft.get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vector=False)
                hog3 = ft.get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vector=False)

            for xb in range(nxsteps):
                for yb in range(nysteps):
                    ypos = yb * cells_per_step
                    xpos = xb * cells_per_step

                    features = []
                    # Extract HOG for this patch
                    if self.hog_feat:
                        hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                        features.extend(np.hstack((hog_feat1, hog_feat2, hog_feat3)))

                    xleft = xpos * self.pix_per_cell
                    ytop = ypos * self.pix_per_cell

                    # Extract the image patch
                    subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                    # Get color features
                    if self.spatial_feat:
                        features.extend(ft.bin_spatial(subimg, size=self.spatial_size))

                    if self.hist_feat:
                        features.extend(ft.color_hist(subimg, nbins=self.hist_bins))

                    # Scale features and make a prediction
                    scaled_features = self.scaler.transform(features)
                    test_prediction = self.svc.predict(scaled_features)

                    if test_prediction == 1:
                        xbox_left = np.int(xleft * scale)
                        ytop_draw = np.int(ytop * scale)
                        win_draw = np.int(window * scale)
                        top_left = (xbox_left, ytop_draw + ystart)
                        bottom_right = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
                        bbox = (top_left,bottom_right)
                        bboxes.append(bbox)

        car_list = self.filterBoxses(img, bboxes, threshold=threshold, debug=debug)

        return car_list

    def filterBoxses(self, img, bboxes, threshold=5, debug=False):
        heatmap_img = self.heatmap(img, bboxes, threshold, debug=debug)
        final_bboxes = self.heatmapToBoxes(heatmap_img, debug=debug)
        return final_bboxes

    def heatmap(self, img, bboxes, threshold=5, debug=False):
        heatmap = np.zeros(img.shape[:2])
        for bbox in bboxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1

        if debug: utils.showImage(heatmap, cmap='hot')
        heatmap[heatmap < threshold] = 0
        if debug: utils.showImage(heatmap, cmap='hot')
        return heatmap

    def heatmapToBoxes(self, heatmap_img, debug=False):
        bboxes = []
        heatmap_regions, cnt_regions = label(heatmap_img)
        if debug:
            utils.showImage(heatmap_regions)
            print(cnt_regions)

        for region_number in range(1,cnt_regions+1):
            nonzero = (heatmap_regions == region_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            bboxes.append(bbox)

        return bboxes

    # Define a function to extract features from a list of images
    # Have this function call bin_spatial() and color_hist()
    def extractFeatures(self, imgs, debug=False):
        # Create a list to append feature vectors to
        last_progress = 0
        features = []
        # Iterate through the list of images
        for i, file in enumerate(imgs):
            file_features = self.extractFeature(file, debug=debug)

            # merge features
            features.append(np.concatenate(file_features))
            progress = int((i * 100) / len(imgs))
            if progress > last_progress:
                print('Feature extraction progress', progress, '%')
                last_progress = progress

        # Return list of feature vectors
        return features

    def extractFeature(self, img, debug=False):

        features = []
        image = utils.loadImage(img)
        image = utils.color_space(image,cspace=self.color_space)
        image = self.normalizeImage(image)

        if self.spatial_feat:
            spatial_features = ft.bin_spatial(image, size=self.spatial_size, debug=debug)
            features.append(spatial_features)

        if self.hist_feat:
            # Apply color_hist()
            hist_features = ft.color_hist(image, nbins=self.hist_bins, debug=debug)
            features.append(hist_features)

        if self.hog_feat:
            hog_features = []
            channels = [self.hog_channel]
            if self.hog_channel == 'ALL':
                channels = range(image.shape[2])

            for channel in channels:
                feature_vec = ft.get_hog_features(
                    image[:, :, channel],
                    self.orient,
                    self.pix_per_cell,
                    self.cell_per_block
                )
                hog_features.append(feature_vec)
            hog_features = np.ravel(hog_features)
            features.append(hog_features)

        # Return list of feature vectors
        return features

    def normalizeImage(self, img):
        return img.astype(np.float32) / 255
        # return (img.astype(np.float32) - 127) / 127 # performs much worst


