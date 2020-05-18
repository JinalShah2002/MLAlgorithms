#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author Jinal Shah

This file will run a test
case for the Regularization Gradient
Descent. 

"""
# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from RegGradLin import LinearRegression

# Getting the Data
PATH = '/Users/jinalshah/Jinal/Github Repos/MLAlgorithms/Data/RegLin.mat'
raw_data = scipy.io.loadmat(PATH)

# Getting X and y
X = raw_data['X']
y = raw_data['y']

# Plotting the data
plt.scatter(X,y,c='red')
plt.xlabel('Change in Water Level')
plt.ylabel('Water Flowing out of the dam')
plt.show()

# Building the regressor
regressor = LinearRegression()
regressor.fit(X,y)