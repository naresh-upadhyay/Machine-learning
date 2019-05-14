#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 11:05:26 2019

@author: king
"""
#import libraries
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

#import dataset 
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

# dummy variable creation

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()

# remove one dummmy variable
x=x[:,1:]

#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# model creation 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

# testing the created model
y_pre=regressor.predict(x_test)
"""
# building the optimal model using backward eliminition (manually)
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,1,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,3,5]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x[:,[0,3]]
regressor_ols=sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()


# Automatic backward elimination using pvalue
import statsmodels.formula.api as sm
def backwardElimination(x1, sl):
    numVars = len(x1[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x1).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x1 = np.delete(x1, j, 1) #np.delete(arr,index,axis) axis=>row =0,column=1
        regressor_OLS.summary()
    return x1
    
SL = 0.05
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
X_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""
# Automatic backward elimination with pvalues and rsaqure
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1) #np.delete(arr,index,axis) axis=>row =0,column=1
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
        regressor_OLS.summary()
    return x

SL = 0.05
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
X_opt = x[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)




























