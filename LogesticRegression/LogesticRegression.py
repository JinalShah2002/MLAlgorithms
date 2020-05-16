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
      self.alpha = 0.01
      self.iterations = 1500
      self.cost = []
      self.iterat = []
    
    # Fit Method
    def fit(self,X,y):
        ones = np.ones((X.shape[0],1))
        self.X = np.concatenate((ones,X),axis=1)
        self.y = y
        self.theta = np.zeros((self.X.shape[1],1))
        
    # Sigmoid function
    def sigmoid(self,z):
        return 1.0/(1+np.exp(-z))
    
    # Hypothesis Function
    def getHypothesis(self):
        z = np.matmul(self.X,self.theta)
        return self.sigmoid(z)

    # Getting the Cost Function
    def getCost(self):  
        sum1 = np.matmul(-self.y.T,np.log(self.getHypothesis()))
        sum2 = np.matmul((1-self.y).T,np.log(1.0-self.getHypothesis()))
        m = len(self.y)
        
        # Final Calculation
        return (1/m) * np.sum(sum1-sum2)
    
    # Setting the Alpha
    def setAlpha(self,a):
        self.alpha = a
        
    # Setting the Iterations
    def setIterations(self,i):
        self.iterations = i
        
    # Getting the Gradient
    def gradient(self):
        h = self.getHypothesis()
        m = len(self.y)
        error = h - self.y
        
        return (1/m) * np.matmul(error.T,self.X).T
    
    # Deploying Gradient Descent
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
        return np.matmul(val,self.theta)
        
    
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