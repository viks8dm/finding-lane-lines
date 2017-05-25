#**Self-Driving Car Engineer Nanodegree** 

##Project-1: Finding Lane Lines on the Road

###Writeup author: Vikalp Mishra
###Date of submission: 23rd Feb 2017

---

**Project Goals**

The goals / steps of this project are the following:

* Make a pipeline that finds lane lines on the road
* Reflect on your work in this report




[//]: # (Image References)

[image1]: ./saved_images/mask_region_of_interest.jpg "region-of-interest mask"
[image2]: ./saved_images/lines_after_HT.jpg "marked lines after Hough Transform line detection"
[image3]: ./saved_images/solidWhiteCurve_lane_identified.jpg "Lane identified, after slope based filtering"
[image4]: ./saved_images/extrapolated_lanes.jpg "Extrapolated lanes after identification"
[image5]: ./saved_images/faint_image_too_many_edges.jpg "opational-challenge video frame"
[image6]: ./saved_images/faint_lane_edges.jpg "opational-challenge video frame-edges"
[image7]: ./saved_images/clear_lane_edges.jpg "test-image frame-edges"

---

### ReflectioN

###1. Description of pipeline.

My pipeline consists of following steps:

1. The code starts with either reading the standalone image or the image from the video-stream and convert it to grayscale using OpenCV module cv2.cvtColor.
2. Gaussian smoothening and Canny algorithm are applied for edge detection. After playing around with the test-images provided as part of the project and some images from the video files, I am using these values for the parameters (kernel-size=5, low-threshold=70, high-threshold=210). In the process of optimizing these parameters it was observed that low-threshold=20 gives better results with the 'extra' or 'challeng' video, however that generates too mane edges, so I went back to using a alue of 70.
3. Thereafter I define a trapezoidal zone as mask or region-of-interest, as shown by blue dotted line in [image1]. The boundaries of the region of interest were selected based on location of camera and lane lines; assumption is that the camera location will remain fixed relative to the car.
![alt text][image1]
4. Following this, Hough Transform was appplied for line detection, as discussed in the lesson videos; and the lines thus identified are used as potential lanes, as shown in [image2]. ![alt text][image2]
5. In [image2] above, one can see that in addition to lane markings, a line on the car in adjoining lane and a line on the road-edge is also marked as a line after Hough Transform, however, these are not lane markings and need to be removed from the selection. To address this issue, I updated the algorithm and select only the lines whose slope-magnitude falls between 0.3 and 0.9. These values are based on relative location of camera with respect to lanes as is in all the test images and videos. The modified lines are marked with red in [image3]. ![alt text][image3]
6. In order to draw a single line on the left and right lanes, I modified the draw_lines() function by separating lines identified above into right and left lines, based on the value of slope for each lines. For example, lines with slope between 0.4 & 0.7 are classified as right-lane-lines while those with slopes between -0.9 & -0.5 are classified as left-lane-lines. The local slope value along with point coordinates [x,y] are used to compute intercept (c) for the equation of a line, y=m*x+c (where, m=slope). This information is further used to calculate average slope and intercept (c) for right and left line sets. These mean slope & intercept values are assumed to be true slope & intercept values for right and left lanes respectively. This true slope & intercept value along with maximum and minimum y-values in the region-of-interest ([image1]) are used to compute the associated max & min x-values for both lanes in the region-of-interest. This extrapolation gives right and left lanes on a typical image, as shown in [image4]. ![alt text][image4]
7. [link to my P1.ipynb:] (http://localhost:8888/notebooks/P1.ipynb?token=bb29412bbd814ea24889284a6568b5a6a806d5dbab1e5096#)


###2. Potential shortcomings with current pipeline

Some shortcomings as I understand are:

* The Gaussian-blurring does not work great when the lanes makerkings are faint or if it leads to identification of too many features as edges; a case with one of the 'optional challenge' video frame, shown in [image5] and [image6]; unlike a test-image, example [image7], where lane marking edges could be clearly extracted. ![alt text][image5]  ![alt text][image6] ![alt text][image7] I tried andMedian blurring worked better, but I could not obtain 100% success with my parameter combinations. 
* Due to same reason as above, failure might be observed when lanes fall in shadows.
* The algorithm identifies and separates lanes based on slope of edges in a region of interest. Hence, it might fail when there are sharp turns.
* The situation will be more challenging if the road is too crowded and lane markings are not easily visible extractable.


###3. Possible improvements
Possible improvements could be:

* to extract edges within certain slope range in [image6] and only look at those in the region of interest
* if clear lane markings are not observed in the region of interest, region can be modified to account for sharp turns. I need to further think about how to modify the region of interest as there can be many scenarios.
* dynamic changes in parameters for image-blurring and Hough-transform can be used to address situations as in 'optional-challenge' video.
