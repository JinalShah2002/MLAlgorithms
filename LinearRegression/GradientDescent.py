#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author Jinal Shah

This file will use Gradient Descent to perform
Linear Regression

Note: For now, this implementation will be
unregularized. I will develop a new regularized
implementation later on

"""
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    
    # Constructor
    def __init__(self):
        self.iterations = 1500
        self.alpha = 0.01
        self.cost = []
        self.iterat = []
    
    # Fit method
    def fit(self,X,y):
        ones = np.ones([X.shape[0],1])
        self.X  = np.concatenate((ones,X),axis=1)
        self.y = y
        self.theta = np.zeros([self.X.shape[1],1])
        
    
    # Getting the Hypothesis
    def hypothesis(self):
       return np.dot(self.X,self.theta)
   
    # Cost Function
    def getCost(self):
        # Setting the starting variables
        m = self.X.shape[0]
        h = self.hypothesis()
        
        # Getting the Error
        error = np.subtract(h,self.y)
        
        # Getting the summation of error squared
        summation = np.sum(np.square(error))
        
        # Final Part
        return (1/(2*m)) * summation
    
    # Getting the Gradient
    def gradient(self):
        # Declaring length and hypothesis variables
        m = self.X.shape[0]
        h = self.hypothesis()
        
        # Declaring the error
        error = h - self.y
        
        summation = np.dot(error.T,self.X).T
        return (1/m) * summation
    
    # Method to set the Iterations
    def setIterations(self,i):
        self.iterations = i
        
    # Method to set Alpha
    def setAlpha(self,a):
        self.alpha = a
    
    
    # Performing Gradient Descent
    def gradientDescent(self):
        
        for i in range(1,self.iterations):
            grad = self.gradient()
            self.theta = np.subtract(self.theta,self.alpha*grad)
            self.cost.append(self.getCost())
            self.iterat.append(i)
            
            # Makes sure the cost is not increasing
            try:
                if self.cost[i] < self.cost[i+1]:
                    return('Gradient Descent is not working properly!')
            except:
                pass
            
    # Predict Method
    def predict(self,*args):
        # Calling gradient descent if needed
        if np.array_equal(self.theta,np.zeros([self.X.shape[1],1])):
            self.gradientDescent()
        # Column of ones
        val = np.ones((len(args[0]),1))
    
        # Combining the features
        for i in range(0,len(args)):
            temp = args[i]
            val = np.concatenate((val,temp),axis=1)
        
        # Getting the predictions(s)
        return np.dot(val,self.theta)
        
    
    # A plot of the Cost Function 
    def plot(self):
        # Calling gradient descent if needed
      if np.array_equal(self.theta,np.zeros([self.X.shape[1],1])):
            self.gradientDescent()
      # Plotting the graph
      plt.plot(self.iterat,self.cost,c='blue')
      plt.xlabel('Number of Iterations')
      plt.ylabel('Cost Function')
      plt.title('Cost Function v.s the Number of Iterations')