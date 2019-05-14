#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 06:25:22 2019

@author: king
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#impoting datasets and making tansetion as list of lists
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
  
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# listing the result
result=list(rules)
