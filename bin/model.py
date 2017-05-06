import os
import csv

# Import driving data (images & speed).  Store in numpy arrays (Keras needs this format).
samples = []

with open('../data-udacity/data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)
#with open('../data-fastest/driving_log.csv') as csvfile:
#  reader = csv.reader(csvfile)
#  for line in reader:
#    samples.append(line)
#with open('../data-red-white-track/driving_log.csv') as csvfile:
#  reader = csv.reader(csvfile)
#  for line in reader:
#    samples.append(line)
#with open('../data/driving_log.csv') as csvfile:
#  reader = csv.reader(csvfile)
#  for line in reader:
#    samples.append(line)
print("line count: ", len(samples))

# Divede the samples into training and validation data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)
print("train_samples count: ", len(train_samples))
print("validation_samples count : ", len(validation_samples))

import cv2
import numpy as np
import sklearn
import random

# The generator which stores the images and angles in numpy arrays in batches
#     reducing the memory footprint
def generator(samples, batch_size=32):
  num_samples = len(samples)
  while 1:
    random.shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      images = []
      angles = []
      steering_correction = 0.2
      for batch_sample in batch_samples:
        # store each of the 3 camera's images (center, right, and left) for training later
        current_path = '../data/IMG/'  
        center_image = cv2.imread(current_path + batch_sample[0].split('\\')[-1])
        left_image   = cv2.imread(current_path + batch_sample[1].split('\\')[-1])
        right_image  = cv2.imread(current_path + batch_sample[2].split('\\')[-1])
        images.append(center_image)
        images.append(left_image)
        images.append(right_image) 
        
        # store the angles used for each of the 3 camera's images
        center_angle = float(batch_sample[3]) #retrieve the actual steering angle for the center image
        # adjust the left & right camera angles to accommodate the camera's offset from the center
        left_angle = center_angle + steering_correction 
        right_angle = center_angle - steering_correction  
        angles.append(center_angle)
        angles.append(left_angle)
        angles.append(right_angle) 

        # store the reverse image and angle for each stored sample
        augmented_images, augmented_angles = [], []
        for image,angle in zip(images, angles):
          augmented_images.append(image) #store the normal image for training
          augmented_angles.append(angle) #store the normal angle for training
          augmented_images.append(cv2.flip(image,1)) #store a vertically-flipped image for training
          augmented_angles.append(angle*-1.0) #store the reversed steering angle for training
      
        X_train = np.array(augmented_images)
        y_train = np.array(augmented_angles)
      yield sklearn.utils.shuffle(X_train, y_train)

# used to compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = 32)
validation_generator = generator(validation_samples, batch_size = 32)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# define the neural network
model = Sequential()
#normalize the data
model.add( Lambda( lambda x: x/255.0 - 0.5, input_shape=(160,320,3) ) )
#remove unnecesary parts of the training images to help reduce noise and thus allow
#  the network to focus on the relevant parts of the image.
model.add( Cropping2D( cropping=((70,25),(0,0)) ) )

# create the nvidia network architecture with slight modification(s)
model.add( Convolution2D(24,5,5,subsample=(2,2),activation="relu") )
model.add( Convolution2D(36,5,5,subsample=(2,2),activation="relu") )
model.add( Convolution2D(48,5,5,subsample=(2,2),activation="relu") )
#model.add( Convolution2D(64,3,3,activation="relu") )
#model.add( Convolution2D(64,3,3,activation="relu") )
model.add( Flatten() )
model.add( Dense(100) )
model.add( Dense(50) )
model.add( Dense(10) )
model.add( Dropout(0.05) ) # add dropout to help reduce overfitting
model.add( Dense(1) )

# compile and training the network
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples*3*2),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
