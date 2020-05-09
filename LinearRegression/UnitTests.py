#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author Jinal Shah

This is the file where I will run 
unit tests 

I have decided to use the already
implemented sklearn's linear regression
class to compare my predictions


"""
# Importing the Libaries
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from GradientDescent import LinearRegression
from sklearn.linear_model import LinearRegression as lin_reg

class TestLinearRegression(unittest.TestCase):
    
    # Cost Function Unit Tests
    def test_cost(self):
        # Getting the Data
        PATH = '/Users/jinalshah/SpiderProjects/ML Algorithm Implementations/Data/Profits.csv'
        data = pd.read_csv(PATH)
        
        # Splitting the Data into X and y
        X = data['Population'].values
        y = data['Profits'].values
        
        # Reshaping X and y into the appropriate shape
        X = X.reshape(97,1)
        y = y.reshape(97,1)
        
       # Creating Regressor
        regressor = LinearRegression()
        regressor.fit(X, y)
        
        # Testing the Cost Function
        self.assertEqual(round(regressor.getCost(),2),32.07)
    
    # Testing the Single Feature Gradient Descent Algorithm
    def test_single_grad(self):
        # Getting the Data
        PATH = '/Users/jinalshah/SpiderProjects/ML Algorithm Implementations/Data/Profits.csv'
        data = pd.read_csv(PATH)
        
        # Splitting the Data into X and y
        X = data['Population'].values
        y = data['Profits'].values
        
        # Reshaping X and y into the appropriate shape
        X = X.reshape(97,1)
        y = y.reshape(97,1)
        
        # Splitting the data into training and testing
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        
       # Creating both regressors
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        regressor_2 = lin_reg()
        regressor_2.fit(X_train,y_train)
    
        # Prediction val
        pop = np.array(15).reshape(1,1)
        
        # Getting both predictions
        grad_predict = regressor.predict(pop)
        lib_predict = regressor_2.predict(pop)
        
        # Testing the Grad Descent Algorithm
        """
        
        While both predictions will not be the same,
        it is safe to say that Gradient Descent works
        as the error between the 2 predictions is small
        
        """
        assert grad_predict - lib_predict <= 0 or (grad_predict - lib_predict >= 0 
                                                   and grad_predict - lib_predict <1)
        
    
if __name__ == '__main__':
    unittest.main()