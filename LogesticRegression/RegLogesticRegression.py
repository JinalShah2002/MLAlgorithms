#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author Jinal Shah

This is my regularized implementation 
of Logestic Regression

"""
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt

class LogesticRegression():
    
   # Constructor
    def __init__(self):
      self.alpha = 0.001
      self.iterations = 400
      self.reg = 1
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
        m = len(self.y)
        sum1 = np.matmul(-self.y.T,np.log(self.getHypothesis()))
        sum2 = np.matmul((1-self.y).T,np.log(1.0-self.getHypothesis()))
        r = self.reg/(2*m) + np.sum(np.power(self.theta, 2))
        
        # Final Calculation
        return (1/m) * np.sum(sum1-sum2) + r
    
    # Setting the Alpha
    def setAlpha(self,a):
        self.alpha = a
        
    # Setting the Iterations
    def setIterations(self,i):
        self.iterations = i
    
    # Setting the Regularization Parameter
    def setReg(self,r):
        self.reg = r
        
    
    def gradient(self):
        h = self.getHypothesis()
        m = len(self.y)
        error = h - self.y
        temp0 = np.array((1/m) * (error[0] * self.X[0]).T).reshape(3,1)
        temp1 = np.array((1/m) * np.matmul(error[1:].T,self.X[1:]).T + self.reg/m * self.theta)
        grad = temp0 + temp1
        return grad
    
    
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
    