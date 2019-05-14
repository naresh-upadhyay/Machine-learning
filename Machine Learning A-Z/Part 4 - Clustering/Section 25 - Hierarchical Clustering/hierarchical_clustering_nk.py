#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:42:07 2019

@author: king
"""


# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

# constructing the dendrogram
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram creation')
plt.xlabel('customers')
plt.ylabel('Euclidean distances')
plt.show()

# hierarchical clustring creation
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)


#ploting each cluster
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c='red',label='careful')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c='green',label='standard')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c='blue',label='target')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c='pink',label='careless')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c='gray',label='sensible')
plt.title('Hierarchical cluster formation')
plt.xlabel('Annual income')
plt.ylabel('Spending')
plt.legend()
plt.show()



