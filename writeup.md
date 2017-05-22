##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


[//]: # (Image References)
[pos_neg]: ./output_images/pos_neg.png
[cls_cnn_infer]: ./output_images/cls_cnn_infer.png
[cls_cnn_train]: ./output_images/cls_cnn_train.png
[dataset_cleanup]: ./output_images/dataset_cleanup.png

[detections1]: ./output_images/detections1.png
[detections2]: ./output_images/detections2.png
[detections3]: ./output_images/detections3.png
[detections4]: ./output_images/detections4.png
[detections5]: ./output_images/detections5.png
[detections6]: ./output_images/detections6.png

[models]: ./output_images/models.png
[labels_map]: ./output_images/labels_map.png
[prediction_map]: ./output_images/prediction_map.png
[project_video_final]: ./project_video_final.mp4



**Vehicle Detection Project**

The goals of the project is to perform vehicle detection in a video stream in a robust and efficient way.
During this project I implemented and tested 2 solutions, using traditional ML with HOG, histogram and spatial feature to train and SVM.
However after spending long time tuning the pipeline and write a quite complex tracker to overcome the limits of SVM I wasn't satisfied with the result,
 mostly because, I knew it was destined to fail miserably in real life conditions, a working pipeline can be be found in previous commits of this git repository (I switch to a NN classifier on the 14th of May) 
So, remembering the video lectures from cs231n on CNN and localization I decided to try to give it a shoot. 
Among the various implementations proposed in the lecture I decided to use pure CNN where the FC layer are removed and 
the classification is done by using a Nx1x1 convolution where N is the number of classes (for more details see https://youtu.be/2xtx-gk3PqY?t=17m7s).
The main reason of using a pure CNN is that the resulting network can be used on images of difference size and the output would scale accordingly, producing a prediction map containing the probability of a class appearing in the given portion of an image. 
The prediction map is then thresholded to keep only high confidence predictions. 
Then each prediction is projected back on the original image using the size of the patches used initially to train the network, 64x64 in my case.
I've tried other approaches, like directly mapping the points between prediction and original image or just simple rescaling.
With the various test I've done, performance wise, the best results are obtained are by slightly blurring the prediction map (Gaussian 3x3), threshold and used the resulting matrix as a scaled down version of the heatmap.
While all the approaches listed above works quite well, however, formally speaking is a more correct to reproject the prediction on the image using the original size of the patch used for training, this is the current implementation.

_I apologies if the current solution still may still contains small traces of the SVM code, but right now I don't have time for further refactoring, I hope the code is clean enough._

### Project structure

- **main.py**  
  Main project file, set train=True to train a new model
- **model.py**  
  Model definition and training.
- **classifier.py**
  Was mostly used with the SVM solution, currently contain methods to produce heatmap and bounding boxes
- **dataset.py**
  Methods to handle the datasets, generators, augmenting, extraction from videos (udacity dataset), etc
- **utils**
  General purpose functions for loading, saving and displaying images, deal with filesystem, etc
- **traker.py**
  Contains `Traker()` and `Detection()` classes used by the pipeline to track vehicles over several frames
  
- **dataset/**
  contains complete datasets
- **datasrouces/**
  contains raw images divided into 2 main directories, vehicles, and non-vehicles. Each category if further organized in subdirs to identify the soruce of the images (ex: GTI, Extra, augmented, flipped, etc )
- **models/**
  Saved keras models
- **videos/**
  Video output folder, final prodcut of the pipeline.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

In my finial implementation, as I explained in the introduction I decided not to use an SVM classifier due to is several limitation. 
However, there are many analogy between a HOG and the first layer of ConvNet, infact, by looking at the filter learned we will find oriented edges, the main difference is that where HOG computes the blindly, the convnet learn only the meaningful ones at training time. 
Further more, the ConvNet has also the ability to aggregate information coming from multiple color channels at once.

###Sliding Window Search

As mention in the introduction, using a ConvNet as described above, give us a sliding window mechanism almost for free as it comes in the form of a convolution, is extremely fast to compute and produce a quite dense output.
The equivalent number of "sliding windows" can be easily obtained by looking at the size of the prediction map, every element of the last layer of the CNN is equivalent to a window.



### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_final.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As mention in the introduction I used the prediction map outputted by the CNN as starting point. 
Compared to the result obtained with the SVM turn out to be much more dense, clean, precise and polarized, however is still not immune from false positive and missing detections.

In the final version of the code I perform the following steps: 
- I use a gentile blur (Gaussian 3x3) over the original prediction map to fill the gaps and reduce noise.
- I then threshold the map to keep only high confidence predictions 
- Reproject all the non-zero predictions, using the original training image size, 64x64, on it on an empty image of the same size of the original producing a heatmap.
- Using `scipy.ndimage.measurements.label()` I regroup adjacent groups of pixels and produce bounding boxes.
- The resulting bounding boxes are then passed over to a Tracker object of filtering an matching (see below)
- The Detections validated by the tracker are then used to produce the green bounding boxes the you seen on the video.

Tracking steps:
- The Tracker object performs a simple validation using the size and position of the box:
  - Opposite lane (no need to take actions)
  - Too small
  - Wrong width/height ratio
- Using the valid boxes I then test them against the detections found in previous frames and select the one that have the best overlapping area using the python library `shapely` 
- If an agreeable match is not found a new Detection object is created and added to the list of detections.
- In any case, the matched detection (or the newly created one) is marked as detected, which increase it's internal counter and new position is stored.
- The list of detection is purged of old detections.
- The list of currently visible detection is return, in case the pass they are considered robust enough (found in a minimum amount of frames)
  
I wrote the Traker for the SVM and I simply adapted (simplified actually) it for the CNN classifier, there are many improvements I would have loved to include but I already had implement this project twice, here a few examples:
- trajectory and position prediction by interpolating using the centroids of the boxes and the frame number
- leverage the rigidity of template matching for a very robust identification of the single vehicle and as a failsafe mechanism, in case a Detection suddently disappear (ex: a bounding box include multiple adjacent vehicles)
  
### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

