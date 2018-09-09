**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/originals.png
[image2]: ./output_images/hog_viz.png
[image3]: ./output_images/hog_features.png
[image4]: ./output_images/tot_raw_features.png
[image5]: ./output_images/tot_scaled_features.png
[image6]: ./output_images/YUV_xfrm.png
[image7]: ./output_images/raw_detection.png
[image8]: ./output_images/heat_maps.png
[image9]: ./output_images/final_detection.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

My HOG feature extraction is implemented in the class method `VehTracker.get_hog_features()`. 
This method is used in `VehTracker.extract_features()` during training and `VehTracker.find_cars()` during execution. 
I first stored all the training images locally in two folders, `/training_data_vehicle` and `/training_data_non_vehicle`, and utilized `glob.glob()` to read in the images. 
In total there are 8792 car training samples and 8968 non-car training samples.
Here is an example of a vehicle and non-vehicle training set:

![alt text][image1]

I explored the variouos color spaces (HSV, RGB, YUV, and YCrCb) and had the best training validation results with YUV, so each image was converted to YUV.
The color conversion is implemented in `VehTracker.convert_color()`.
I then calculated the HOG features for each image and each color channel with `orientations = 10`, `pixels_per_cell = 16`, and `cells_per_block = 2`.
I discovered that increasing `pixels_per_cell` dramatically increated compute speed which was one of my major practical hurdles in this project, but hardly affected accuracy.
Below is an example of HOG calculated on the YUV color space:

![alt text][image2]

The features are concatenated to form the whole feature vector:

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I based my hog `orientation` parameter on the lecture videos that recommended 9 orientations, and increased slightly for hopes of improved accuracy. 
As mentioned above I also increased `pixels_per_cell` in order to increase compute time without hurting accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Originally I tried training my SVM with spatial binning features, color transform features, and HOG features, but this resulted in an extremely slow pipeline (about 3 min to train and over 40 min to process the whole video).
After inspecting the histograms it became apparent that the spatial binning features were not representing any obvious uniqueness of the vehicles over the non vehicles.
In addition, with over 3000 features for each 64x64 image, this feature set was hurting compute time more than it was helping classification. 
Similarly, I tested my pipeline without the color histogram features and found almost no impact to accuracy but a noticable improvement to runtime, so I removed those features.
In the end my feature vector consisted only of HOG on a YUV color transform.

I trained a linear SVM and utilized `GridSearchCV()` when training to automatically optimize the parameters.
I tested with the following parameters `svm_params = {'kernel':['linear'], 'C':[0.001, 0.1, 1, 10, 100]}` and found that `C = 0.1` was optimal.
Note that originally I included the rbf kernel but soon discovered that the training time was just too long to be practical.
I also utilized a `StandardScaler()` to normalize the features. 
Here is an example of normalization of ALL possible features (spatial, color, and HOG):

![alt text][image4]

![alt text][image5]

My final SVM had a test accuracy of ~97% and trained in 243s.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search as explained in the lectures in the method `VehTracker.find_cars` with some changes.
Instead of searching over just one scale of the original image, I looped over a list of scale factors and a list of search regions.
My final implementation used the scale factors `scale_list = (1.0, 1.0, 1.5, 1.5, 2.0, 3.0, 3.5)` and the search regions `ystop_list = (450, 490, 500, 525, 550, 600, 650)`.
Each scale factor corresponds to the same element ystop (and a ystart of 400 for all).
This helped my pipeline to better identify cars on the horizon.
Next I changed `cells_per_step = 1` so that there was more overlap thereby giving my classifier more changes to catch the vehicle in a position that it would recognize.
Otherwise, I used the default hyperparameters `ystart = 400` and others mentioned in section 1.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I started off testing my pipeline with all possible features (spatial, color, and HOG) but soon discovered that spatial and color were costing more compute time than they were worth.
Below I will cover my implemented pipeline on a set of samples:

![alt text][image1]

First convert to YUV:

![alt text][image6]

Next calculate HOG on all channles of the YUV image:

![alt text][image3]

The resultant raw detection boxes:

![alt text][image7]

A thresholded heatmap:

![alt text][image8]

The final detection boxes:

![alt text][image9]

Note that in the video implementation, the heat map is actually a rolling sum which helps to further filter out erroneous detections.
When processing the video I used these tuning parameters for my rolling filter `heatmap_flt_size = 5` and `heat_threshold = 25`.
That is, the previous 5 frame's heatmaps are summed and only locations with heat of 25 or greater are considered part of a bounding box.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://youtu.be/0OYqUcmBOgc)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Furthermore I also implemented a rolling sum heatmap that could be used to filter out erroneous detections (described above).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest struggle I had with this project was finding a way to iterate over various hyperparameters within a practical train and prediciton time. 
This has further practical implementations for a real pipeline which would need to run in real time, which this implementaiton would not.
At times my implementations would take over 30 min to process the project video.

The second area of the video which I never quite got right is near the middle when the white car gets relatively far away and it seems that the whole frame becomes lighter (perhaps due to a shift in orientation).
I believe my training data simply had insufficient samples for this particular orientation of the car so in order to improve that, I ought to capture more data of the vehicle in such an orientation.
Furthermore, it might be that I just dont have quite the right scaling factors to "catch" the car in an orientation that the SVM can recognize. I could never find this magic number.
Finally, I suspect that the light change could be addressed by a second set of features based on a different color transform, but I dont think it would be practical to implement.
As it stands, the current pipeline already runs nowhere near the required realtime speed, so additional large feature vectors are unrealistic.
