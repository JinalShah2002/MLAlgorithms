#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author Jinal Shah

This file will be used to compare the
various algorithms for Linear Regression


"""
# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GradientDescent import LinearRegression as grad
from NormalEquation import LinearRegression as norm
from sklearn.model_selection import train_test_split

# Getting the data
PATH = '/Users/jinalshah/SpiderProjects/ML Algorithm Implementations/Data/Profits.csv'
raw_data = pd.read_csv(PATH)

# Splitting the data into X and y
X = raw_data.copy().drop('Profits',axis=1).values
y = raw_data.copy().drop('Population',axis=1).values

# Plotting the data for insights
plt.scatter(X,y,c='Red')
plt.xlabel('Population')
plt.ylabel('Profits')
plt.title('Profits v.s Population')
plt.show()

# Splitting the Data into training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Building the Linear Regression model
grad_regressor = grad()
norm_regressor = norm()
grad_regressor.fit(X_train, y_train)
norm_regressor.fit(X_train, y_train)


# Plotting the final hypothesis
plt.scatter(X_train,y_train,c='red')
plt.plot(X_train,grad_regressor.predict(X_train),c='blue')
plt.plot(X_train,norm_regressor.predict(X_train),c='green')
plt.title('Profits v.s Population')
plt.xlabel('Population')
plt.ylabel('Profits')
plt.show()

"""

As you can see, the lines 
are pretty much the same!


"""