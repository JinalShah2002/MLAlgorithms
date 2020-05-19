#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author Jinal Shah


This file will house the Neural Network Class.
I will be building a 3 layer (1 input, 1 hidden, and 1 output)
Neural Network. I am implementing this for understanding and 
for fun! 

Final note, all activation functions will be sigmoid

"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    
    # Constructor
    def __init__(self):
        pass
    
    # Fit Method
    def fit(self,X,y):
        # Declaring my biases
        self.b1 = np.ones((X.shape[0],1))
        self.b2 = np.ones((X.shape[0],1))
      
        self.X = np.concatenate((self.b1,X),axis=1)
        self.y = y
        
        # Declaring my weights
        self.weights1 = np.random.random((self.X.shape[1],2))
        self.weights2 = np.random.random((3,1))
    
    # Forward Propagation Method
    def forwardPropagation(self):
        # Hidden Layer 1
        z1 = np.matmul(self.X,self.weights1)
        self.a2 = self.sigmoid(z1)
        
        # Output
        a3 = np.concatenate((self.b2,self.a2),axis=1)
        self.output = np.matmul(a3,self.weights2)
        self.output = self.sigmoid(self.output)
        
        return self.output
    
    # Backward Propagation
    def backwardPropagation(self):
        
        pass
   
    # Sigmoid Function Method
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    
    
    # Putting it together
    def predict(self):
        pass