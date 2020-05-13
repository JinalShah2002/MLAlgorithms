#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author Jinal Shah

This file will be used to implement
unregularized Logestic Regression

Reminder, logestic regression is a 
popular classification algorithm

"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt

class LogesticRegression():
    
    # Constructor
    def __init__(self):
      pass
    
    # Fit Method
    def fit(self,X,y):
        ones = np.ones((X.shape[0],1))
        self.X = np.concatenate((ones,X),axis=1)
        self.y = y
        self.theta = np.zeros((self.X.shape[1],1))
    
    # Sigmoid function
    def sigmoid(self,z):
        bottom = 1 + np.exp(-z)
        return 1/bottom
    
    # Hypothesis Function
    def getHypothesis(self):
        z = np.matmul(self.X,self.theta)
        return self.sigmoid(z)

    # Getting the Cost Function
    def getCost(self):
        sum1 = np.matmul(-self.y.T,np.log(self.getHypothesis()))
        sum2 = np.matmul((1-self.y).T,np.log(1-self.getHypothesis()))
        m = len(self.y)
        
        # Final Calculation
        return (1/m) * np.sum(sum1-sum2)
    
    # Getting the Gradient
    def gradient(self):
        h = self.getHypothesis()
        m = self.X.shape[0]
        error = h - self.y
        
        return (1/m) * np.matmul(error.T,self.X).T