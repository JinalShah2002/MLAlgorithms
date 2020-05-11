#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author Jinal Shah

This file will the file where I implement the 
Normal Equation (Ordinary Least Squares) for 
Linear Regression. 

This implementation will be unregularization, 
I will implement a regularized version in the
future

"""
# Importing libraries
import numpy as np

class LinearRegression():
    
    # Constructor
    def __init__(self):
        pass
    
    # fit method
    def fit(self,X,y):
        # Column of Ones
        ones = np.ones([X.shape[0],1])
        self.X = np.concatenate((ones,X),axis=1)
        self.y = y
        self.theta = np.zeros([self.X.shape[1],1])
    
    # Getting the thetas
    def normalEquation(self):
        inverse = np.linalg.inv(np.matmul(self.X.T,self.X))
        inverseAndX = np.matmul(inverse,self.X.T)
        self.theta = np.matmul(inverseAndX,self.y)
    
    # Predict method
    def predict(self,*args):
        # Calling Normal Equation if needed
        if np.array_equal(self.theta,np.zeros([self.X.shape[1],1])):
            self.normalEquation()
        # Column of ones
        val = np.ones((len(args[0]),1))
    
        # Combining the features
        for i in range(0,len(args)):
            temp = args[i]
            val = np.concatenate((val,temp),axis=1)
        
        # Getting the predictions(s)
        return np.matmul(val,self.theta)        