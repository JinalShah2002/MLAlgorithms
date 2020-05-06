#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author Jinal Shah

Suppose you are the CEO of a restaurant franchise and are 
considering different cities for opening a new outlet. 
The chain already has trucks in various cities and you have data for 
profits and populations from the cities.You would like to use this data to 
help you select which city to expand to next.

This is the Gradient Descent tester. Since I am only looking to test
Gradient Descent and make sure it works, the given problem has already
been determined to be a Linear Regression problem. For future problems,
the user will have to go through the Machine Learning project development
steps!

"""
# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GradientDescent import LinearRegression
from sklearn.model_selection import train_test_split

# Getting the data
PATH = '/Users/jinalshah/SpiderProjects/ML Algorithm Implementations/Data/Profits.csv'
raw_data = pd.read_csv(PATH)

# Splitting the data into X and y
X = raw_data.copy().drop('Profits',axis=1).values
y = raw_data.copy().drop('Population',axis=1).values

"""
Note: Since this class is only for the sole purpose
of testing the Gradient Descent implementation, the 
data has already been preprocessed so no preprocessing
is needed

However, for future data sets, you will need to 
preprocess the data. If needed, you will need to 
apply feature scaling because gradient descent
doesn't do it for you!

"""

# Plotting the data for insights
plt.scatter(X,y,c='Red')
plt.xlabel('Population')
plt.ylabel('Profits')
plt.title('Profits v.s Population')
plt.show()

# Splitting the Data into training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Building the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# Predicting the testing set
y_pred = regressor.predict(X_test)

# Plotting the final hypothesis
plt.scatter(X_train,y_train,c='red')
plt.plot(X_train,regressor.predict(X_train),c='blue')
plt.title('Profits v.s Population')
plt.xlabel('Population')
plt.ylabel('Profits')
plt.show()

# Plotting the Cost Function 
regressor.plot()
