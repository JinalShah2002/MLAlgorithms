#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author Jinal Shah

This file will run Linear Regression. In
contrast to my past algorithms, I will be
using a highly advanced optimization 
algorithm from Scipy. 

I am implementing this just to understand
the difference between the results. Also,
it is fun!


"""
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class LinearRegression():
    
    # Constructor
    def __init__(self):
        self.iterations = 400
    
    # Fit method
    def fit(self,X,y):
        ones = np.ones([X.shape[0],1])
        self.X  = np.concatenate((ones,X),axis=1)
        self.y = y
        self.theta = np.zeros([self.X.shape[1],1])
        
    
    # Getting the Hypothesis
    def hypothesis(self):
       return np.matmul(self.X,self.theta)
   
    # Cost Function
    def getCost(self,y):
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
    def gradient(self,y):
        # Declaring length and hypothesis variables
        m = self.X.shape[0]
        h = self.hypothesis()
        
        # Declaring the error
        error = h - self.y
        
        summation = np.matmul(error.T,self.X).T
        return (1/m) * summation
    
    # Method to set Iterations
    def setIterations(self,i):
        self.iterations = i
    
    # Optimization
    def optimize(self):
        options = {
            'maxiter':self.iterations
            }
        res = minimize(fun=self.getCost,x0=self.theta,jac=self.gradient,options=options)
       
     
    # Predict Method
    def predict(self,*args):
        # Calling advanced optimization if needed
        if np.array_equal(self.theta,np.zeros([self.X.shape[1],1])):
            self.optimize()
        # Column of ones
        val = np.ones((len(args[0]),1))
    
        # Combining the features
        for i in range(0,len(args)):
            temp = args[i]
            val = np.concatenate((val,temp),axis=1)
       
        # Getting the predictions(s)
        return np.matmul(val,self.theta)