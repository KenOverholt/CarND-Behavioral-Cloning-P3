# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center-lane-original.jpg "Center-lane driving"
[image3]: ./examples/recovery-from-right-1.jpg "Recovery from right"
[image4]: ./examples/recovery-from-right-2.jpg "Recovery from right"
[image5]: ./examples/reverse-direction-1.jpg "Reverse driving"
[image6]: ./examples/reverse-direction-2.jpg "Reverse driving"
[image7]: ./examples/red-white1.jpg "Track with red/white edges"
[image8]: ./examples/red-white2.jpg "Track with red/white edges"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
After training with the current data set, the first autonomous drive complete 1+ full lap successfully.  Later runs occasionally got stuck but some have also successfully completed a full lap so, of course, the model could be improved.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is the NVIDIA architecture consisting of 5 convolutional layers from 24 to 64 followed by a flattening layer and then 4 dense layers with sized 100, 50, 10, and l (model.py lines 51-60) 

The model includes RELU activations for the convolutional layers to introduce nonlinearity (code lines 51-55), and the data is normalized in the model using a Keras lambda layer (code line 45). 

#### 2. Attempts to reduce overfitting in the model

Epochs were kept down to 5 (code line 63) keeping the validation loss from rising in order to reduce overfitting. 

The model was trained and validated on different data sets (80/20 split) to ensure that the model was not overfitting (code line 63). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 62).

#### 4. Appropriate training data

I chose training data attempting to keep the vehicle driving on the road. I tried multiple variations of training data. My initial set was created by just attempting to drive as best I could to get around the track while staying on it.  I also tried with my initial set and cropping the images to reduce the noise.

I tried adding a lap swerving from side to side attempting to give the network examples of how to correct when veering off course.  I added a lap of data running in the opposite direction.  I used the Unity engine to modify track 1 by adding white & red stripes to the edge of the track and I gathered a data set on that custom track.

My final set was 1.5 laps around the track swerving from side to side not getting to wild in the areas with no edges on the track.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try use a proven architecture and then, if necessary, modify a layer or two.  I ended up not needing to modify it.

My first step was to try LeNet because that worked will in the sign classification and is a well-known architecture.

In order to gauge how well the model was working, I split my image and steering angle data 80/20 into a training and validation set. I found that my first model had a higher mean squared error on the training set and a slightly higher mean squared error on the validation (0.0737) set but was coming down for 5 epochs at which point it started to increase.  The increase in validation error implies that the model is overfitting so throughout my tests, I kept epochs to a count just below the point when validation started increasing.

Cropping the data improved the loss slightly (0.0634) but not enough so I next switched to the NVIDIA architecture.

NVIDIA produced loss on the order of 0.0051 which is significantly lower but the car still didn't stay on the track all the way around until I changed my traning data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 51-60) is the NVIDIA architecture consisting of 5 convolutional layers from 24 to 64 followed by a flattening layer and then 4 dense layers with sized 100, 50, 10, and l (model.py lines 51-60) 

The model includes RELU activations for the convolutional layers to introduce nonlinearity (code lines 51-55), and the data is normalized in the model using a Keras lambda layer (code line 45).  It produces one value which is the steering measurement.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap on track one using center lane driving as best I could (I wasn't that great a simulator driver but improved over the course of the proeject). I steered using the keyboard.  Here is an example image of center lane driving:

![alt text][image2]

I later recorded the vehicle swerving from the left side and right sides of the road back to center so that the vehicle would learn to recover when it veered off-course.  I started using the mouse to steer at this point and used it for the remaining of the data gathering sessions. These images show what a recovery looks like starting from:

![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would provide double the data showing steering in the opposite direction since the track is mostly left turns. I also drove around the track in the opposite direction to get samples of reverse turn driving but more variety since I would be controling rather than just exact opposite values and images.  Here are samples of driving in the opposite direction:

![alt text][image5]
![alt text][image6]

I noticed my car would sometimes drive into the red & white edges so I used the Unity engine to modify track one by adding red & white edges all around the cement curb.  Here is a sample of those images:

![alt text][image7]
![alt text][image8]

For each training session, I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs varied but was 4 or 5 as evidenced by the increase in loss from that point on. I used an adam optimizer so that manually training the learning rate wasn't necessary.
