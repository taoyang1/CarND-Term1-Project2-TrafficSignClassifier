## **Finding Lane Lines on the Road** 

### Tao Yang, March 22, 2017


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_out/out_solidWhiteCurve.jpg "solidWhiteCurve"
[image2]: ./test_images_out/out_solidWhiteRight.jpg "solidWhiteRight"
[image3]: ./test_images_out/out_solidYellowCurve.jpg "solidYellowCurve"
[image4]: ./test_images_out/out_solidYellowCurve2.jpg "solidYellowCurve2"
[image5]: ./test_images_out/out_solidYellowLeft.jpg "solidYellowLeft"
[image6]: ./test_images_out/out_whiteCarLaneSwitch.jpg "whiteCarLaneSwitch"


---

### Reflection

#### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of the following 6 steps:

1. Convert the images to grayscale;
2. Apply Gaussian smoothing to blur the image;
3. Apply Canny transformation to the image;
4. Mask the image with a four side polygon;
5. Run Hough transformation on detected edges to extract lines;
6. Draw connected lines on the raw images.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by the following steps:

1. For each line segment in the returned list of Hough transformation, calculate their slope by m = (y2-y1)/(x2-x1)
2. Test if m is positive or negative. If m > 0, then this line belong to the right line, if m < 0, the left line.
3. Count all the line segments that belong to the left line, denoted as left_count, similarly for right_count.
4. For all line segments belong to the left line set, cacluate their mid point x_left_mid by averaging all the x coordinates. Repeat this steps for x_right_mid, y_left_mid and y_right_mid. 
5. Calculate the average slope for the left line and the right line by taking the average of all the slopes, denote them by m_left and m_right.
6. Finally extrapolate the line from (x_left_mid, y_left_mid) and (x_right_mid, y_right_mid) to both top and bottom of the lane. The y-coordinate of the top and bottom are selected to be img.shape[0] and 350. Their x-coordinate can be calcuated using m_left and m_right.

The results of running my pipeline on the test images yield the following output:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

#### 2. Identify potential shortcomings with your current pipeline


i. One potential shortcoming would be what would happen when there is curved lane. 

ii. Another shortcoming could be when there are horizontal lines cross the line.

iii. Another shortcoming is the manual and tedious process of tuning parameters for Canny, Hough, etc. It will be good to automate such process via machine learning.


#### 3. Suggest possible improvements to your pipeline

A possible improvement would be to in the draw line function, instead of simplying using the sign of m (slope) to determine weather it's left line or right line, use kmeans to detect two groups of slopes. This will automatically discard those horizontal lines and avoid the manual process.

