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
        self.X =X
        self.y = y
        