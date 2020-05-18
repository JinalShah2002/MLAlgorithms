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

class NeuralNetwork():
    
    # Constructor
    def __init__(self):
        # Declaring my biases
        self.b1 = 1
        self.b2 = 1
        
    
    # Fit Method
    def fit(self,X,y):
        self.X = X
        self.y = y
        
        # Declaring my weights
        self.weights1 = np.random.random(3,1)
        self.weights2 = np.random.random(3,1)
        self.weights3 = np.random.random(3,1)
    
    # Sigmoid Function Method
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))