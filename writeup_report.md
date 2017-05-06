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
[image7]: ./examples/red-white-1.jpg "Track with red/white edges"
[image8]: ./examples/red-white-2.jpg "Track with red/white edges"

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
After training with various network architectures and multiple sets of simulation data that I generated, I ended up using the Udacity-provided data with a modified architecture in order to complete a full lap around track one.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is a modified version of the NVIDIA architecture consisting of 3 convolutional layers from 24 to 48 followed by a flattening layer and then 3 dense layers with sizes 100, 50, and 10.  Next is a dropout later with 5% dropout followed by a dense layer size l (model.py lines 97-107) 

The model includes RELU activations for the convolutional layers to introduce nonlinearity (code lines 97-99) and the data is normalized in the model using a Keras lambda layer (code line 91). 

#### 2. Attempts to reduce overfitting in the model

I kept epochs low at 5 (code line 113).  This kept the validation loss from rising in order to reduce overfitting.  I also introduced a 5% dropout layer before the final dense layer to help reduce overfitting as well. 

The model was trained and validated on different data sets (80/20 split) to ensure that the model was not overfitting (code line 27). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 110).

#### 4. Appropriate training data

I chose training data attempting to keep the vehicle driving on the road. I tried multiple variations of training data.

My initial set was created by just attempting to drive as best I could to get around the track while staying on it.  At one point I started cropping the images to reduce the noise.  The network was able to focus on the parts of the image that contributed to steering.

Next I tried adding a lap swerving from side to side attempting to give the network examples of how to correct when veering off course.  I also added a lap of data running in the opposite direction.  For another set of data, I used the Unity engine to modify track one by adding white & red stripes to the edge of the track and I gathered a data set on that custom track.

I had a working model consisting of 1.5 laps around the track mildly swerving from side to side with some driving down the center and, at the spots where the cement curb disappeared, I swerved away from the dirt edge making sure to provide samples indicating dirt edges are to be avoided.  However, when I introduced the dropout layer, the car would not stay on the track for an entire lap.

I ended up using the data set provided by Unity.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try use a proven architecture and then, if necessary, modify a layer or two.  I ended up removing the the final 2 convolution layers and adding a droput layer.

My first step was to try LeNet because that worked well in the sign classification project and is a well-known architecture.

In order to gauge how well the model was working, I split my image and steering angle data 80/20 into a training and validation set. I found that my first model had a higher loss on the training set and a slightly higher loss on the validation (0.0737) set but was coming down for 5 epochs at which point it started to increase.  The increase in validation error implies that the model is overfitting so, throughout my tests, I kept epochs to a count just below the point when validation loss started increasing.

Cropping the data improved the loss slightly (0.0634) but not enough so I next switched to the NVIDIA architecture.

NVIDIA produced loss on the order of 0.0051 which is significantly lower but the car still didn't stay on the track all the way around until I changed my traning data.  However, once I added the dropout layer, the car could not stay on the track.

I tried experimenting with combinations of the 4 data sets I had gathered with various modfications to the NVIDIA architecture but none of them would keep the car on the track.  Using all the data sets caused the system to run out of memory so I introduced a generator which broke the training up into small batches reducing the memory footprint and allowing the large data set to run.

My final approach was to use the Unity-provided data set on a pared-down version of the NVIDIA architecture.  At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 97-107) is a modification of the NVIDIA architecture.  My archicture consists of 3 convolutional layers from 24 to 48 followed by a flattening layer and then 4 dense layers with sizes 100, 50, 10, and l (model.py lines 103-107).  There is a 5% droput layer (code line 106) before the last dense layer.

The model includes RELU activations for the convolutional layers to introduce nonlinearity (code lines 97-99), and the data is normalized in the model using a Keras lambda layer (code line 91).  The network produces one value which is the steering angle required to keep the car on the track.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input               | 160x320x3 color jpg image
| Normalization       | Adjust values to between -0.5 and 0.5
| Cropping         		| remove top 75 and bottom 25 rows for 65x320x3 color jpg image | 
| Convolution 5x5     | 2x2 stride, valid padding 	              |
| RELU					      |	activation						                    |
| Convolution 5x5	    | 2x2 stride, valid padding      						|
| RELU		            | activation					                      |
| Convolution 5x5			| 2x2 stride, valid padding    							|
| RELU                | activation                                |
|	Flatten					    |			          									|
|	Dense     					| outputs 100												|
| Dense               | outputs 50        |
| Dense               | outputs 10 |
| Dropout 5%          |         |
| Dense               | outputs 1                                 |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap on track one using center lane driving as best I could (I wasn't that great a simulator driver but improved over the course of the proeject). I steered using the keyboard.  Here is an example image of center lane driving:

![alt text][image2]

I later recorded the vehicle swerving from the left side and right sides of the road back to center so that the vehicle would learn to recover when it veered off-course.  I started using the mouse to steer at this point and used it for the remaining of the data gathering sessions. These images show what a recovery looks like starting from the right side:

![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would provide double the data showing steering in the opposite direction since the track is mostly left turns. I also drove around the track in the opposite direction to get samples of reverse turn driving but with more variation since I was controling rather than just exact opposite values and images.  Here are samples of driving in the opposite direction:

![alt text][image5]
![alt text][image6]

Another augmentation I tried was to use the images from the left and right camera.  I used that data and modified the angles slightly allowing the network to use these images as if they were in the center.

I noticed my car would sometimes drive into the red & white edges so I used the Unity engine to modify track one by adding red & white edges all around the cement curb.  Here is a sample of those images:

![alt text][image7]
![alt text][image8]

For each training session, I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs varied but was 4 or 5 as evidenced by the increase in loss from that point on. I used an adam optimizer so that manually training the learning rate wasn't necessary.
