#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 15:02:18 2019

@author: king
"""


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#implementing UCB algo
import math
d = 10 #no. of ads
N = 10000 #no. of tests or peoples
total_reward = 0
sum_of_reward = [0]*d
number_of_selections = [0]*d
ads_selected = []
for n in range(0,N):
    average_reward = 0
    delta_i = 0
    max_upper_bound = 0
    ad = 0
    for j in range(0,d):
        upper_bound = 0
        if number_of_selections[j] > 0 :
            average_reward = sum_of_reward[j]/number_of_selections[j]
            delta_i = math.sqrt((3/2)*math.log(n+1)/number_of_selections[j])
            upper_bound = average_reward+delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound :
            max_upper_bound = upper_bound
            ad = j
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sum_of_reward[ad] = sum_of_reward[ad] + reward
    total_reward = total_reward + reward
    
# Visualizing using histogram
plt.hist(ads_selected)
plt.title('Upper Confidence Bound histogram')
plt.xlabel('Number of ads')
plt.ylabel('Number of times ads clicked')
plt.show()
            