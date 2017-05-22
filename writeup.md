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

[crash1]: ./output_images/crash1.png
[crash2]: ./output_images/crash2.png



**Vehicle Detection Project**

The goals of the project is to perform vehicle detection in a video stream in a robust and efficient way.
  
During this project I implemented and tested 2 solutions, using traditional ML with HOG, histogram and spatial feature to train and SVM.  

However after spending long time tuning the pipeline and write a quite complex tracker to overcome the limits of SVM I wasn't satisfied with the result, mostly because, I knew it was destined to fail miserably in real life conditions, a working pipeline can be be found in previous commits of this git repository (I switch to a NN classifier on the 14th of May)   
so, remembering the video lectures from cs231n on CNN and localization I decided to try to give it a shoot.
  
Among the various implementations proposed in the lecture I decided to model my architecture after OverFeat 2013 (http://www.image-net.org/challenges/LSVRC/2013/slides/overfeat_ilsvrc2013.pdf) using pure CNN where the FC layer are removed and 
the classification is done by using a Nx1x1 convolution where N is the number of classes, for more details see (https://youtu.be/2xtx-gk3PqY?t=17m7s).  

The main reason of using a pure CNN is that the resulting network can be used on images of difference size and the output would scale accordingly, producing a prediction map containing the probability of a class appearing in the given portion of an image.   
The prediction map is then thresholded to keep only high confidence predictions.  
 
Then each prediction is projected back on the original image using the size of the patches used initially to train the network, 64x64 in my case.
I've tried other approaches, like directly mapping the points between prediction and original image or just simple rescaling.

With the various test I've done, performance wise, the best results are obtained are by slightly blurring the prediction map (Gaussian 3x3), threshold and used the resulting matrix as a scaled down version of the heatmap.

While all the approaches listed above works quite well, however, formally speaking is a more correct to reproject the prediction on the image using the original size of the patch used for training, this is the current implementation.

_I apologies if the current solution still contains small traces of the SVM code, but right now I don't have time for further refactoring, I hope the code is clean enough._

Model at train
![cls_cnn_train]

Model at prediction
![cls_cnn_infer]


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
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

In my finial implementation, as I explained in the introduction I decided not to use an SVM classifier due to is several limitation. 
However, there are many analogy between a HOG and the first layer of ConvNet, in fact, by looking at the filter learned we will find oriented edges, the main difference is that where HOG computes the blindly, the convnet learn only the meaningful ones at training time.   
Further more, the ConvNet has also the ability to aggregate information coming from multiple color channels at once.  

### Sliding Window Search

As mention in the introduction, using a ConvNet as described above, give us a sliding window mechanism almost for free as it comes in the form of a convolution, is extremely fast to compute and produce a quite dense output.  
The equivalent number of "sliding windows" can be easily obtained by looking at the size of the prediction map (29x153), every element of the last layer of the CNN is equivalent to a window.  

![prediction_map]

The very first and probably the most important factor to improve the accuracy of this solution is to act on the dataset.  
In order to do so I've tried several approaches, first of all, I parsed the udacity datasets (CrowdAI e Autti) and extracted all the images labeled cars with the idea of after parsing each frame, to blackout the cropped area and slice random images for the negative examples.     
I soon realized, buy looking at the result that many of on my negative examples where still containing cars or part of cars that were not annotated.  
So I decided instead to use the provided dataset and augment the negative examples to balance the 100.000 images produced.  
After many attempts and long hours of training I realized that the resulting model was having much more false positives, unexpectedly.  
I then decide to as on-the-fly augmentation directly in the generator, the resulting performances were still very good, even if the training and the validation accuracy where very good at training time, when tested on the video where not perfoming as expected.    
So I dropped every augmentation or additional dataset and stick just with the provided one, I just have added horizontal flip of the car, the show to enlarge the area of detection.  
Only very recently, by accident, while I was doublechecking the augmented images to make sure that everything went well I started to understand the problem with my dataset, the negative examples are full of images of cars, here a few examples put together by checking the first 200 images of one of the provided datasets (non-vechicle, Extra)    

![dataset_cleanup]

This gave me the idea of doublecheck the dataset, I realized the most of the problems I have with detection can be solved by carefully select and balance the example of cars in the dataset having equal numbers of:
- car from the side
- car from the back
- car from the front
- car with light on
- car with light off
For the categories above also equal examples of cars of different colors and of course, no cars or part of cars in my negative examples.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_final.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

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
  
### Here are six frames with the debug fully activated, showing every step on the processing in a single image

__Legend:__
- Top left: prediction map as outputted from the classifier
- Top right: heatmap
- Red Boxes: Bounding boxes coming from `scipy.ndimage.measurements.label()`
- Green boxes: Confirmed detections, produced by the `Tracker()` (the only one visible in the final output)

![detections1]  
![detections2]  
![detections4]
![detections5]
![detections6]
![detections3]
_(I like this last one particularly because it show how much my NN likes little red dots as they resemble car stop signs, I guess)_

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Starting from the the first implementation, using SVM, I found staggering the amount of parameters (and the combination of them) that I needed to try to just be able to extract features,
this mixed with the long time of processing of the video and the random appearance of false positive and false negative randomly along the frames that was forcing me to test every change on most of the video to make sure I didn't break something while trying to fix something else, I must say have been a quite frustrating experience.
I resorted to add PyGame to my project to play a loud sound to mark the end of each phase (or a crash)
A side, but annoying issue was (due to my limited experience with python) was that due to the size of the computed features
I soon reached the limit of the filesize of pickle that I was originally using to store the computed feature, so I started to try other methods, from cvs, to h5py and finally I found the numpy offers a quite simple mechanism, but have several limitations.
Another issue is when I tried to feed to the scaler the whole dataset of computed feature including all the images coming from the udacity dataset (CrowdAI + Attti) about 200.000 images (including the negative)

![crash1]
![crash2]

I then decided to switch to a NN implementation, it went so much faster, both the implementation ad the training,
however I have been facing several issues in picking the correct architecture, luckily I got an enlightening conversation with a fellow student of my cohort that choose a OverFeat neural network as well   
for this project, so I tired out his architecture and it was working so much better then mine.  

So far the main difference was that his model performed a pulling operation only at the very end, right before the classifier.  
So I tried to change it and modified over and over again in terms of layer size, number of layers, activation function, colorspaces, I got many models training extremely fast with great accuracy, but so far performing quite poorly on the actual video.  
One example above all really made me think, is when the car was overtaking, my network was completely blind while his could see it, shortly after, as soon as the back of the car was visible all a sudden my network spotted it.  

I got to the conclusion that the pooling operation only preserves the most common traits of an object and discard everything else used for matching (front and back) while his network instead managed to preserve the information intact through the layers.    
I bet that with the properly curated dataset (see sliding window section) I would be able to train a much more efficient models without loosing any robustness.  

Also, I've notice how choosing a color space over another make my classifier (both SVM and CNN) become affectionate to a color channel, for example with YCrCb it "really likes" black and white thing in the image where with RGB it got "affectionate" to the red channel, so I made a bit of research and I found this very nice paper https://arxiv.org/ftp/arxiv/papers/1511/1511.01064.pdf and I really would like to explore this idea in my next projects.    