#### **Traffic Sign Recognition** 

## Tao Yang, 04/24/2017

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/barplot_old.png "Bar chart for original training datasets"
[image3_1]: ./plots/barplot_new.png "Bar chart for augmented datasets"
[image3_2]: ./plots/loss_curve.png "loss curve as a function of epochs"
[image4]: ./test_images/test0.jpg "Traffic Sign 1"
[image5]: ./test_images/test1.jpg "Traffic Sign 2"
[image6]: ./test_images/test2.jpg "Traffic Sign 3"
[image7]: ./test_images/test3.jpg "Traffic Sign 4"
[image8]: ./test_images/test4.jpg "Traffic Sign 5"
[image9]: ./plots/softmax_prob.png "bar chart of the top 5 softmax prob"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/taoyang1/CarND-Term1-Project2-TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: 32*32
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.

![bar chart of the distribution of the original training dataset ][image1]

Note that the dataset is not balanced, i.e., some class (i.e., class 2) has more than 10 times data than other classes (i.e., class 0). Therefore, we need data augmentation for better result.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces the input size by three fold.

As a last step, I normalized the image data because LeNet works better for normalized data, i.e., zero mean with small variance.

I decided to generate additional data because the imbalance distribution of training dataset among different classes.

To add more data to the the data set, I used the following techniques because it gives more data for those rare cases:

1. For class with less than 500 counts, generate three rotated images with rotation angles between -10 and 10 degrees.
2. For class with more than 500 but less than 1000 counts, generate on rotated copy with rotation angles between -10 and 10 degrees.
3. For all data, generate one additional translated copy with translation in -4 and 4 pixels (exclude 0). Dataset is divided into three batches for faster processing.


The difference between the original data set and the augmented data set is the following:

1. New data set has 114056 training data vs 34799 in the old dataset. This is roughly 3.5 times more.
2. The distribution of augmented dataset.
![Training data distribution among different classes][image3_1]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Greyscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| TANH					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x	    | 1x1 stride, valid padding, outputs 10x10x16 	|   
| TANH					|												|
| Max pooling			| 2x2 stride,  outputs 5x5x16 					|
| Fully connected		| 400 inputs, 120 outputs, with dropout = 0.7   |
| TANH					|												|
| Fully connected		| 120 inputs, 84 outputs, with dropout = 0.7    |
| TANH					|												|
| Fully connected		| 84 inputs, 43 outputs                         |
| Softmax				| logits         								| 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used trial and error to choose different hyperparameters by checking the validation accuracy as well as the loss curve as a function of epochs. 

The following final hyperparameters are chosen for this projects:
* learning rate = 0.02
* Epochs = 15
* Batch size = 128
* dropout = 0.7

![Loss curve as a function of epochs][image3_2]


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.981
* validation set accuracy of 0.934
* test set accuracy of 0.921

Some notes:
* I use the LeNet as the starting point because of its relative complexity.
* The learning rate and number of epochs are found to be important parameters to tune. They are tuned by the guidance of loss curve. If the loss is slow to going down, then we should be increase the learning rate. If otherwise it's oscillating, then a smaller learning rate should be tried.
* Dropout is used to avoid overfitting and make the model more robust.
* The model reaches 0.95 validation accuracty and 0.924 test accuracy, which is pretty good.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it's not in the training dataset. 
The rest images are in the training dataset. However, their resolution might not be good and could be difficult for the algorithm to identify them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 40 km/ h      		| 50 km/h   									| 
| 50 km/ h     			| 80 km/ h 										|
| 80 km/ h				| 80 km/ h										|
| stop sign	      		| stop sign 					 				|
| Slippery Road			| dangerours curve to the right					|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. This compares a lot worse to the accuracy on the test set of 0.92, which may suggest the model is overfitting to the training dataset. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is relatively sure that this is a Speed limit (50km/h) (probability of 0.9), while it's really a speed limit 40 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .90         			| Speed limit (50km/h)   						| 
| .02     				| Speed limit (60km/h) 							|
| .06					| Speed limit (80km/h)							|
| .00	      			| Speed limit (30km/h)				 			|
| .00				    | No passing for vehicles over 3.5 metric tons  |


For the second image, the model is relatively sure that this is a Speed limit (800km/h) (probability of 0.78), while it's really a speed limit 50 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .78         			| Speed limit (80km/h)   						| 
| .21     				| Speed limit (60km/h)							|
| .00					| Speed limit (50km/h)							|
| .00	      			| Speed limit (100km/h)				 			|
| .00				    | End of speed limit (80km/h)                   |

For the third image, the model is relatively sure that this is a Speed limit (80km/h) (probability of 0.9), and the test image is indeed a speed limit 80 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .90         			| Speed limit (80km/h)   						| 
| .10     				| Speed limit (60km/h) 							|
| .00					| Speed limit (50km/h)							|
| .00	      			| Speed limit (20km/h)				 			|
| .00				    | End of speed limit (80km/h)                   |

For the fourth image, the model is very sure that this is a stop sign (probability of 1.0), and the test image is indeed a stop sign The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop   						                | 
| .00     				| No entry 							            |
| .00					| Go straight or right							|
| .00	      			| Turn right ahead				 			    |
| .00				    | No vehicles                                   |

For the fifth image, the model is relatively sure that this is a Dangerous curve to the right sign (probability of 0.83), while it's really a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .83         			| Dangerous curve to the right   				| 
| .08     				| Slippery road							        |
| .05					| Bicycles crossing							    |
| .02	      			| Road narrows on the right			 			|
| .01				    | Wild animals crossing                         |


Bar chart of the top 5 softmax probability:

![bar chart of the top 5 softmax prob][image9]