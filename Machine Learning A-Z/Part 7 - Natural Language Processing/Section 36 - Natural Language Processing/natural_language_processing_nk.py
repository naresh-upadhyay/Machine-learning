#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:48:45 2019

@author: king
Text cleaning
"""
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)

# remove unnecessary words
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# making matrix of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# importing the classifier code for predicting the review (here for example Random forest)

#spliting the data into training set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0) 

# feature scaling (no need already in range o/1)
# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators = 10, criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

#testing of classification model
y_pred=classifier.predict(x_test)

#creating confusion matrix to check how many prediction are correct or not
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

