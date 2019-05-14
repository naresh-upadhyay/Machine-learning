#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 22:19:41 2019

@author: king
"""

#importing labraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  #import

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values #line , column(except last column)
y=dataset.iloc[:,2].values 

# simple linear regressor
from sklearn.linear_model import LinearRegression
lin_reg1=LinearRegression()
lin_reg1.fit(x,y)

# polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=8)
x_poly=poly.fit_transform(x)
lin_reg2=LinearRegression().fit(x_poly,y)

#testing our prediction
y_pred=lin_reg2.predict(x_poly)

#ploting the graph for simple
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg1.predict(x),color='blue')
plt.title('simple regressor predicted vs actual')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

#ploting graph of polynomial regressor
x_grid=np.arange(min(x),max(x),0.05) # using grid we split all intermediate number in 0.05 size partiiton
x_grid=x_grid.reshape(len(x_grid),1) # reshape (no.of row ,no. of column)
plt.scatter(x,y,color='red')
plt.plot(x_grid,lin_reg2.predict(poly.fit_transform(x_grid)),color='blue')
plt.title('Polynomial regressor predicted vs actual')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()

# prediction using simple regressor model
lin_reg1.predict([[6.5]])

#prediction using polynomial regressor model 
lin_reg2.predict(poly.fit_transform([[13]]))
