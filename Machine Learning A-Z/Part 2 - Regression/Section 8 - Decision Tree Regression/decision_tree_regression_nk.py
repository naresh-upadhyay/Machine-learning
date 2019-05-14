#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 08:00:54 2019

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

# decision tree class importing from treee library
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

#testing regressosr model
y_pred=regressor.predict([[6.5]])

# Decision tree ploting (it works on averaging of a fixed intervel (or spliting))
x_grid=np.arange(min(x),max(x),0.01) #spliting the intervel
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title('actual vs predicted (Decision tree)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()