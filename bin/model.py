import csv
import cv2
import numpy as np

# Import driving data (images & speed).  Store in numpy arrays (Keras needs this format).
lines = []
with open('../data-fastest/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
print(line)

# read in training data as is
images = []
measurements = []
for line in lines:
  source_path = line[0]
  #store the name of the center camera's image
  filename = source_path.split('\\')[-1]
  current_path = '../data-fastest/IMG/' + filename
  image = cv2.imread(current_path) #retrieve the center camera's image
  images.append(image)
  measurement = float(line[3]) #retrieve the steering adjustment
  measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image,measurement in zip(images, measurements):
  augmented_images.append(image) #store the normal image for training
  augmented_measurements.append(measurement) #store the normal measurement for training
  augmented_images.append(cv2.flip(image,1)) #store a vertically-flipped image for training
  augmented_measurements.append(measurement*-1.0) #store the reversed steering measurement for training
  
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
#print("X_train.shape: ", X_train.shape) #debug statement needed when the incorrect path was used above

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#start the neural network
model = Sequential()
#normalize the data
model.add( Lambda( lambda x: x/255.0 - 0.5, input_shape=(160,320,3) ) )
#remove unnecesary parts of the training images to help reduce noise and thus allow
#  the network to focus on the relevant parts of the image.
model.add( Cropping2D( cropping=((70,25),(0,0)) ) )

# create the nvidia network
model.add( Convolution2D(24,5,5,subsample=(2,2),activation="relu") )
model.add( Convolution2D(36,5,5,subsample=(2,2),activation="relu") )
model.add( Convolution2D(48,5,5,subsample=(2,2),activation="relu") )
model.add( Convolution2D(64,3,3,activation="relu") )
model.add( Convolution2D(64,3,3,activation="relu") )
model.add( Dropout( 0.05) )
model.add( Flatten() )
model.add( Dense(100) )
model.add( Dense(50) )
model.add( Dense(10) )
model.add( Dense(1) )

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
