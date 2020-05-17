#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author Jinal Shah

This file will serve as 
the test case for Regularized
Logestic Regression


"""
# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RegLogesticRegression import LogesticRegression
from sklearn.model_selection import train_test_split

# Getting the Data
PATH = '/Users/jinalshah/Jinal/Github Repos/MLAlgorithms/Data/Microchips.csv'
raw_data = pd.read_csv(PATH)

# Splitting Data into X and y
X = raw_data.copy().drop('Result',axis=1).values
y = raw_data.copy()['Result'].values

# Reshaping
X = X.reshape(118,2)
y = y.reshape(118,1)

# Plotting the data
plt.scatter(raw_data['Microchip Test 1'],raw_data['Microchip Test 2'],c=raw_data['Result'])
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()

# Splitting data into training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Building the Regressor
regressor = LogesticRegression()
regressor.fit(X,y)

# Plotting Reg Gradient Descent
regressor.plot()