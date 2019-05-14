#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 20:43:56 2019

@author: king
"""
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

# searching optimal number of clustes using cluster sum of squares
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('ELbow detection')
plt.xlabel('no. of clusters')
plt.ylabel('value of wcss')
plt.show()

#optimal value of cluster then plot clusters
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmean=kmeans.fit_predict(x)

#ploting each cluster
plt.scatter(x[y_kmean==0,0],x[y_kmean==0,1],s=100,c='red',label='careful')
plt.scatter(x[y_kmean==1,0],x[y_kmean==1,1],s=100,c='green',label='standard')
plt.scatter(x[y_kmean==2,0],x[y_kmean==2,1],s=100,c='blue',label='target')
plt.scatter(x[y_kmean==3,0],x[y_kmean==3,1],s=100,c='pink',label='careless')
plt.scatter(x[y_kmean==4,0],x[y_kmean==4,1],s=100,c='gray',label='sensible')
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='centroids')
plt.title('k-means cluster formation')
plt.xlabel('Annual income')
plt.ylabel('Spending')
plt.legend()
plt.show()



