# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#importing labraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  #import

#import dataset
dataset = pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values #line , column(except last column)
y=dataset.iloc[:,3].values 

#taking care of missing data (using sklearn preprocessing library)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = "mean", axis = 0) #ctrl + I (inspect)
imputer = imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()
y=LabelEncoder().fit_transform(y)

#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling(putting values in a fixed range like b/w -1 and 1)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

