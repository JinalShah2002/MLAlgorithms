#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author Jinal Shah

This file will implement 
Regularized Linear Regression, using
the Gradient Descent Algorithm

"""
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    
    # Constructor
    def __init__(self):
        self.iterations = 400
        self.alpha = 0.00001
        self.reg = 0
        self.cost = []
        self.iterat = []
    
    # Fit method
    def fit(self,X,y):
        ones = np.ones([X.shape[0],1])
        self.X  = np.concatenate((ones,X),axis=1)
        self.y = y
        self.theta = np.ones([self.X.shape[1],1])
        
    
    # Getting the Hypothesis
    def hypothesis(self):
       return np.matmul(self.X,self.theta)
   
    # Cost Function
    def getCost(self):
        # Setting the starting variables
        m = self.X.shape[0]
        h = self.hypothesis()
        
        # Getting the Error
        error = np.subtract(h,self.y)
        
        # Getting the summation of error squared
        summation = np.sum(np.square(error))
        
        # Regularization Term
        r = self.reg/(2*m) * np.sum(np.power(self.theta, 2))
        
        # Final Part
        return (1/(2*m)) * summation + r
    
    # Getting the Gradient
    def gradient(self):
        # Declaring length and hypothesis variables
        m = self.X.shape[0]
        h = self.hypothesis()
        
        # Declaring the error
        error = h - self.y
    
        temp0 = np.array((1/m) * (error[0] * self.X[0]).T).reshape(2,1)
        temp1 = np.array((1/m) * np.matmul(error[1:].T,self.X[1:]).T + self.reg/m * self.theta)
        grad = temp0 + temp1
        
        return grad
    
    # Method to set the Iterations
    def setIterations(self,i):
        self.iterations = i
        
    # Method to set Alpha
    def setAlpha(self,a):
        self.alpha = a
        
    # Method to get theta(s)
    def getTheta(self):
        return self.theta
       
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
        if np.array_equal(self.theta,np.ones([self.X.shape[1],1])):
            self.gradientDescent()
        # Column of ones
        val = np.ones((len(args[0]),1))
    
        # Combining the features
        for i in range(0,len(args)):
            temp = args[i]
            val = np.concatenate((val,temp),axis=1)
       
        # Getting the predictions(s)
        return np.matmul(val,self.theta)
        
    
    # A plot of the Cost Function 
    def plot(self):
        # Calling gradient descent if needed
      if np.array_equal(self.theta,np.ones([self.X.shape[1],1])):
            self.gradientDescent()
      # Plotting the graph
      plt.plot(self.iterat,self.cost,c='blue')
      plt.xlabel('Number of Iterations')
      plt.ylabel('Cost Function')
      plt.title('Cost Function v.s the Number of Iterations')
