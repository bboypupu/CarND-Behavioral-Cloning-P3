# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia_architecture.png "Model Visualization"
[image2]: ./examples/center_normal.jpg "Original Image"
[image3]: ./examples/center_recovery.jpg "Recovery Image"
[image4]: ./examples/flip_center.jpg "Flipped Image"
[image5]: ./examples/flip_recovery.jpg "Flipped Recovery Image"
[image6]: ./examples/crop_center.jpg "Cropped Image"
[image7]: ./examples/crop_recovery.jpg "Cropped Recovery Image"
[image8]: ./examples/loss_default.png "Loss Default"
[image9]: ./examples/loss_new.png "Loss New"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter and 3x3 filter sizes and depths between 24 and 64 (model.py lines 106-128) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 109). 

#### 2. Attempts to reduce overfitting in the model

At first I used 2x2 maxpooling layer between convolution layers and dropout layer with keep-probability set to 60% in order to reduce overfitting (model.py lines 21). However, the result in autonomous mode was not good enough (it could finish the lap, but it sometimes touch the edges (especially after crossing the bridge). Meanwhile, I observed that there's space for the loss fo validation. Finally, I deleted the 2x2 maxpooling layer and set the probability in dropout layer to 80%.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 137).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and backward driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

While the training time in this project was very long and so as the penalty of discovering bad model, the overall strategy for deriving a model architecture was to reference the NVIDIA architecture (only added a dropout layer). The model was mainly used for autonomous driving as mentioned in the lecture, and it's really robust in my case. 

Before feeding the data into the model, I adopted the preprocessing procedure: normalization, cropping and flipping horizontally to all three images taken at the same time (center, left, right with steering angle correction of 0.2).

In the first training attempt, I used the default training data provided in the lesson. However, the result wasn't good enough to finish one round. Therefore, I added my own training data, which consisted of 2 laps central, 1 lap backward and 2 laps with recovery movements. I fed the new data to the first model to come up with a new one, and the result was satisfying.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 106-125) consisted of 3 convolutional layers with 5x5 filters, 2 convolutional layers with 3x3 filters, and followed by 3 fully-connected layers and 1 output layer. 

Here is a visualization of the architecture (note: the figure was taken from NVDIA website)

![alt text][image1]

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
lambda_2 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_2 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 1, 33, 64)         36928     
_________________________________________________________________
dropout_2 (Dropout)          (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 100)               211300    
_________________________________________________________________
dense_6 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_7 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 11        
_________________________________________________________________
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to correct the driving curve while reaching the edges of the road. These image show what a recovery looks like:

![alt text][image3]


To augment the data set, I also do a backward lap, and also flipped images from 3 different angles thinking that this would make the situation more complete and eliminate the problem of under fitting. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

On the other hand, the images feeding to the model should be cropped to only the road pattern left, in order to eliminate the noise caused by the tree and sky in the background. The preprocessed image should be like this: 

![alt text][image6]
![alt text][image7]


After the collection process, I had 18162 number of data points. I then preprocessed flipping horizontally, to get 36324 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was just set to 3 while the loss didn't change so much after that. I used an adam optimizer so that manually training the learning rate wasn't necessary. The diagram of the loss in training as shown:

![alt text][image8]
![alt text][image9]
