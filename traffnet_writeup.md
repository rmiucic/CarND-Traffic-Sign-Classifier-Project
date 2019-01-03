# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./ref/sample_input.png "Sample Input"
[image2]: ./ref/img_gray_scld.png "Orig Gray Scaled"
[image3]: ./ref/original_rotated_blured.png "Original Rotated Blured"
[image4]: ./ref/learn_curve.png "Learning Curve"
[image5]: ./ref/sample_count.png "Input Count"
[image6]: ./new_img_png/1.png "Traffic Sign 1"
[image7]: ./new_img_png/2.png "Traffic Sign 2"
[image8]: ./new_img_png/3.png "Traffic Sign 3"
[image9]: ./new_img_png/4.png "Traffic Sign 4"
[image10]: ./new_img_png/5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43

|ClassId |                                          SignName | Count   |
|:---------------------:|:-------------------------:|:------------------------:| 
| 0 |                              Speed limit (20km/h)  |  180   |
| 1 |                              Speed limit (30km/h)  | 1980   |
| 2 |                              Speed limit (50km/h)  | 2010   |
| 3 |                              Speed limit (60km/h)  | 1260   |
| 4 |                              Speed limit (70km/h)  | 1770   |
| 5 |                              Speed limit (80km/h)  | 1650   |
| 6 |                       End of speed limit (80km/h)  |  360   |
| 7 |                             Speed limit (100km/h)  | 1290   |
| 8 |                             Speed limit (120km/h)  | 1260   |
| 9 |                                        No passing  | 1320   |
|10 |      No passing for vehicles over 3.5 metric tons  | 1800   |
|11 |             Right-of-way at the next intersection  | 1170   |
|12 |                                     Priority road  | 1890   |
|13 |                                             Yield  | 1920   |
|14 |                                              Stop  |  690   |
|15 |                                       No vehicles  |  540   |
|16 |          Vehicles over 3.5 metric tons prohibited  |  360   |
|17 |                                          No entry  |  990   |
|18 |                                   General caution  | 1080   |
|19 |                       Dangerous curve to the left  |  180   |
|20 |                      Dangerous curve to the right  |  300   |
|21 |                                      Double curve  |  270   |
|22 |                                        Bumpy road  |  330   |
|23 |                                     Slippery road  |  450   |
|24 |                         Road narrows on the right  |  240   |
|25 |                                         Road work  | 1350   |
|26 |                                   Traffic signals  |  540   |
|27 |                                       Pedestrians  |  210   |
|28 |                                 Children crossing  |  480   |
|29 |                                 Bicycles crossing  |  240   |
|30 |                                Beware of ice/snow  |  390   |
|31 |                             Wild animals crossing  |  690   |
|32 |               End of all speed and passing limits  |  210   |
|33 |                                  Turn right ahead  |  599   |
|34 |                                   Turn left ahead  |  360   |
|35 |                                        Ahead only  | 1080   |
|36 |                              Go straight or right  |  330   |
|37 |                               Go straight or left  |  180   |
|38 |                                        Keep right  | 1860   |
|39 |                                         Keep left  |  270   |
|40 |                              Roundabout mandatory  |  300   |
|41 |                                 End of no passing  |  210   |
|42 | End of no passing by vehicles over 3.5 metric ...  |  210   |

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data look like. The first row shows random images from of ClassId =0, the second row  shows random images from of ClassId =1, and so on.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Process Description 

As a first step, I decided to convert the images to grayscale.

Here is an example of a traffic sign original input image, after grayscaling, and scalling (normalized image).

![alt text][image2]

I normalized the image data because values need to be centered around 0 in order to work neural network in the nonliear domain. 

I decided to generate additional data because my initial learning was not satisfactory.

To add more data to the the data set, I used the following techniques on the origial training set: 1) randomly rotated between (-15,15)  degrees and 2) blured images.
X_train_n.shape is  (34799, 32, 32, 1)
X_train_r.shape is  (34799, 32, 32, 1)
X_train_b.shape is  (34799, 32, 32, 1)
X_train_z.shape is  (34799, 32, 32, 1)
X_train_s.shape is  (139196, 32, 32, 1)
 normalized + rotated + blured + zoomed in
 final training set is the combination of all, concatinated from 4 datasets above.

I did this to increase the number of training data set and to make the network more generic. 

Here is an example of an original image and an augmented images:

![alt text][image3]

#### 2. Final model architecture description 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		                | 32x32x1 grayscale image   						| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	        | 2x2 stride, Valid padding, outputs 14x14x6		|
| Convolution 5x5	        | 1x1 stride, Valid padding, outputs 10x10x16  	|
| RELU					|												|
| Max pooling	      	        | 2x2 stride, Valid padding, outputs 5x5x16		|
| Flatten				| Output = 400 									|
| Fully connected		| Output = 120 									|
| RELU					|												|
| Fully connected		| Output = 84 									|
| RELU					|												|
| Fully connected		| Output = 43 									|
| Softmax				| Output = 43     									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the following hyper parameters

|   Parameter       		| Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Learning Rate         		                | 0.001   				| 
| loss_operation 					| cross_entropy 			|
| Optimizer						| AdamOptimizer		|
| Number of Epochs				| 40					|
| Batch Size						| 128					|

Here is how validation and training learning looked like:

![alt text][image4]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
**training_accuracy 0.999669530734
validation_accuracy 0.951473923173
test_accuracy 0.940380047572
**

If a well known architecture was chosen:
I choose to use LeNet architecture discussed in the class.  The original LeNet was classifying images into 10 classes. I  modified the network to output 43 classes of the traffic signes. The network never saw test set and it had accuracy better than 92%.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
predicted:  [17, 18, 12, 4, 13]
true value: [17, 18, 12, 4, 13]

| Image			        	|     Prediction			| 
|:---------------------:|:---------------------------------------------:| 
| No entry    			| No entry 			| 
| General caution    	| General caution		|
| Priority road			| Priority road			|
| Speed limit (70km/h)	| Speed limit (70km/h)	|
| Yield				| Yield				|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the  images, the model is sure that all prediceted images are what it predicted (probability of close to 1). The top five soft max probabilities are shown below:

INFO:tensorflow:Restoring parameters from ./lenet
TopKV2(values=array([[  1.00000000e+00,   4.65406658e-10,   9.84643547e-11,
          1.05274079e-12,   4.23483397e-15],
       [  1.00000000e+00,   8.50612119e-21,   5.38590546e-28,
          2.53392985e-30,   7.45917988e-37],
       [  1.00000000e+00,   7.07970651e-17,   3.67898240e-28,
          9.91775605e-30,   3.67035580e-31],
       [  1.00000000e+00,   4.17193339e-14,   1.06732633e-19,
          6.30756564e-25,   5.99929058e-30],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00]], dtype=float32), indices=array([[17, 12, 33, 14, 41],
       [18, 26, 27, 15, 25],
       [12, 35, 28, 26, 41],
       [ 4,  0,  1,  5, 25],
       [13,  0,  1,  2,  3]], dtype=int32))