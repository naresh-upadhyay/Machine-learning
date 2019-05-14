#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 01:00:07 2019

@author: king
"""


#importing labraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  #import

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values #line , column(except last column)
y=dataset.iloc[:,2:3].values 

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)

# svr regression
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

# testing prediction by transforming and inverse transforming
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

#ploting the graph for simple
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('simple regressor predicted vs actual')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
