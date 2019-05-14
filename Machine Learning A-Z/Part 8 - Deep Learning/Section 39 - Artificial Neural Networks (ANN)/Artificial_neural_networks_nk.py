#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:12:57 2019

@author: king
"""

# importing important libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
x[:,2] = LabelEncoder().fit_transform(x[:,2])
x[:,1] = LabelEncoder().fit_transform(x[:,1])
onehotencoder = OneHotEncoder(categorical_features =[1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:]

#spliting the data into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0) 

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_y.fit_transform(x_test)

# importing keras library and packeage with its models
import keras
from keras.models import Sequential
from keras.layers import Dense

# creating layers of artificial neural network
classifier = Sequential()

# creating first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu', input_dim = 11 ))

# creating second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu'))


# creating third hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu'))

#creating final layer
classifier.add(Dense(output_dim = 1, init = 'uniform' , activation = 'sigmoid'))

#compiling the ANN that is created above
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Training the model created using ANN
classifier.fit(x_train, y_train, batch_size = 10 , nb_epoch = 20)


#testing of classification model
y_pred=classifier.predict(x_test)
y_pred = ( y_pred > 0.5 )

#creating confusion matrix to check how many prediction are correct or not
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

