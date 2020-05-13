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

"""
# Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from LogesticRegression import LogesticRegression
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

# Building Regressor
regressor = LogesticRegression()
regressor.fit(X,y)