#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author Jinal Shah

This file will serve as the tester file 
for Un-regularized Gradient Descent for 
Logestic Regression

In this test case, we will using 2 Exam scores
to determine whether or not a student obtains 
admission at the given university


Like for the past datasets, this has already
been preprocessed and identified as a classification
problem. For different data sets, the author will 
have to follow the Machine Learning Path!

Key Takeaway: Gradient Descent Requires 
FEATURE SCALING. Make sure to evaluate 
the features and scale if needed!

"""
# Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from LogesticRegression import LogesticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Getting the Data
PATH = '/Users/jinalshah/Jinal/Github Repos/MLAlgorithms/Data/CollegeAdmins.csv'
raw_data = pd.read_csv(PATH)

# Splitting into X and y
X = raw_data.copy().drop('College Admission Decision',axis=1).values
y = raw_data['College Admission Decision'].values

# Splitting Data into X and y 
X = raw_data.copy().drop('College Admission Decision',axis=1).values
y = raw_data['College Admission Decision'].values

# Reshaping
X = X.reshape(100,2)
y = y.reshape(100,1)

# Plotting the data
plt.scatter(raw_data['Exam 1 Score'],raw_data['Exam 2 Score'],c=raw_data['College Admission Decision'])
plt.title('How Scores of 2 exams impact college admission for a certain college')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.show()


# Building Regressor & Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
regressor = LogesticRegression()
regressor.fit(X_scaled,y)
regressor.plot()


# Predictions
scaler_2 = StandardScaler()
exam_1 = np.array(45).reshape(1,1)
exam_2 = np.array(85).reshape(1,1)
combined = np.concatenate((exam_1,exam_2),axis=1)
print(regressor.predict(scaler_2.fit_transform(combined)))