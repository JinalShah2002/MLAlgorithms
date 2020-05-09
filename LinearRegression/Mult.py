#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author Jinal Shah

This file will serve as a test file for 
Gradient Descent when the data has multiple features.

As said before, this is only a test case file. Not a 
full end to end Machine Learning project! I am only
testing my implemented Gradient Descent algorithm!

"""
# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GradientDescent import LinearRegression
from sklearn.model_selection import train_test_split

# Getting the dataset
PATH = '/Users/jinalshah/SpiderProjects/ML Algorithm Implementations/Data/MultVarLin.csv'
raw_data = pd.read_csv(PATH)

# Splitting data into X and y
X = raw_data.copy().drop('Price',axis=1).values
y = raw_data.copy()['Price'].values

# Transforming X and y into numpy arrays
X = X.reshape(47,2)
y = y.reshape(47,1)

# Split Training and Testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling 
# It is important to note that Gradient Descent
# requires the data's features to be scaled

def featureScaling(data):
    top = data - data.mean()
    return top/data.std()

X_scaled = featureScaling(X_train)

# Building the model
regressor = LinearRegression()
regressor.fit(X_scaled,y_train)

# Plotting Gradient Descent to make sure it is working
regressor.setAlpha(0.001)
regressor.plot()

# Prediction
size = np.array(1650).reshape(1,1)
rooms = np.array(3).reshape(1,1)
pred = regressor.predict(size,rooms)
print(pred)