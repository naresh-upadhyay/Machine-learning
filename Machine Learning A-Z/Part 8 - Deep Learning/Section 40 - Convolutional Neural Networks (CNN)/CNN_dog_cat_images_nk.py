#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 01:52:55 2019

@author: king
"""
# importing necessary library and function and classes
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense

# start creating layers using sequential function (you can use graph method )
classifier = Sequential()

# step1- Convolution layer (no. of feature mape and size of feature detector 2D (32,3,3)) & relu for reactifier linear unit
classifier.add(Convolution2D(32,3,3, input_shape = ( 64 , 64, 3), activation = 'relu')) #input_shape = (pixel size 2D,RGB) image size we can incresse it for more accuracy (but on GPU) 

# step2- layer Pooling layer (take pool size (2x2) to preserve the image )
classifier.add(MaxPool2D(pool_size = (2,2)))


# second Convolution layer (to increse accuracy of output) not need of input_shape again 
classifier.add(Convolution2D(32,3,3, activation = 'relu')) #keras manage it automatically(we can increase n->2*n (eg. 32->2*32) on each addtion of convolutional layer)
classifier.add(MaxPool2D(pool_size = (2,2)))

#step3- layer Flattering layer 
classifier.add(Flatten())

#step4- layer (Full connection) CNN (generally take output_dim value not too much approx 100 and good if it in 2's power)
classifier.add(Dense(output_dim = 128 , activation = 'relu'))# take automatically init = 'uniform'
#classifier.add(Dense(output_dim = 128 , activation = 'relu'))# take more hidden layer in CNN for more accuracy output 
#classifier.add(Dense(output_dim = 128 , activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))# for binary output sigmoid(probabilistic) otherwise softmax function

#compile the CNN (optimizer here is stochastic gradient decent)
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics= ['accuracy'])

# fitting this CNN to our problem
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, #like feature scalling (scale fixels b/w 0-255)
                                   shear_range=0.2,#geometrical tranformationn in pixel (shifting by a fixed amount)
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',                                                
                                                 target_size=(64, 64), #size of input image
                                                 batch_size=32,#no. of baches in one time execution (one forward execution)
                                                 class_mode='binary') #no. of output

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,                         
                         steps_per_epoch=8000,#no of images in training set(no. of cases pass in one execution)
                         epochs=25,#no. of time relaxating the CNN
                         validation_data=test_set,
                         validation_steps=2000)#no. of images in test set
