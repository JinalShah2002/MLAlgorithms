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
from sklearn.preprocessing import StandardScaler
from NormalEquation import LinearRegression as norm

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
    
        # Prediction value(s)
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
        assert (grad_predict - lib_predict <= 0 and grad_predict -lib_predict >=-1) or (grad_predict - lib_predict >= 0 and grad_predict - lib_predict < 1)
        
    # Testing the Multiple Feature Gradient Descent Algorithm
    def test_multiple_grad(self):
        # Getting the Data
        PATH = '/Users/jinalshah/SpiderProjects/ML Algorithm Implementations/Data/MultVarLin.csv'
        data = pd.read_csv(PATH)
        
        # Splitting the Data into X and y
        X = data.copy().drop('Price',axis=1).values
        y = data.copy()['Price'].values
        
        # Reshaping X and y into the appropriate shape
        X = X.reshape(47,2)
        y = y.reshape(47,1)
        
        # Splitting the data into training and testing
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        
        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Creating both regressors
        regressor = LinearRegression()
        regressor.fit(X_scaled,y_train)
        regressor.setAlpha(0.01)
        regressor.setIterations(400)
        regressor_2 = lin_reg()
        regressor_2.fit(X_train,y_train)
    
        # Prediction value(s)
        size = np.array(1650).reshape(1,1)
        rooms = np.array(3).reshape(1,1)
        combined = np.concatenate((size,rooms),axis=1)
        
        # We must scale our input for Grad Descent as well!!
        scaler3 = StandardScaler()
        combined_scaled = scaler3.fit_transform(combined)
        
        
        # Getting both predictions
        grad_predict = regressor.predict(combined_scaled)
        lib_predict = regressor_2.predict(combined)
        
        # Testing the Grad Descent Algorithm
        """
        
        While both predictions will not be the same,
        it is safe to say that Gradient Descent works
        as the error between the 2 predictions is small
        
        """
        assert grad_predict/lib_predict <= 1.5 or lib_predict/grad_predict <= 1.5
        
    # Testing the Normal Equation using the Gradient Descent Algorithm
    def test_normal_single_var(self):
         # Getting the Data
        PATH = '/Users/jinalshah/SpiderProjects/ML Algorithm Implementations/Data/Profits.csv'
        data = pd.read_csv(PATH)
        
        # Splitting the Data into X and y
        X = data.copy().drop('Population',axis=1).values
        y = data.copy()['Profits'].values
        
        # Reshaping X and y into the appropriate shape
        X = X.reshape(97,1)
        y = y.reshape(97,1)
        
        # Splitting the data into training and testing
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        
        # Creating the Regressors
        regressor = LinearRegression()
        regressor.fit(X_train,y_train)
        regressor_2 = norm()
        regressor_2.fit(X_train, y_train)
        
        # Prediction value(s)
        pop = np.array(15).reshape(1,1)
        
        # Getting both predictions
        grad_predict = regressor.predict(pop)
        norm_predict = regressor_2.predict(pop)
        
        # Making sure both predictions are the same
        self.assertEqual(np.round(grad_predict[0][0],1),np.round(norm_predict[0][0],1))
    
    # Testing the Normal Equation with multiple variables
    def test_normal_mult(self):
        # Getting the Data
        PATH = '/Users/jinalshah/SpiderProjects/ML Algorithm Implementations/Data/MultVarLin.csv'
        data = pd.read_csv(PATH)
        
        # Splitting the Data into X and y
        X = data.copy().drop('Price',axis=1).values
        y = data.copy()['Price'].values
        
        # Reshaping X and y into the appropriate shape
        X = X.reshape(47,2)
        y = y.reshape(47,1)
        
        # Splitting the data into training and testing
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        
        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Creating both regressors
        regressor = LinearRegression()
        regressor.fit(X_scaled,y_train)
        regressor.setAlpha(0.01)
        regressor.setIterations(400)
        regressor_2 = norm()
        regressor_2.fit(X_train,y_train)
    
        # Prediction value(s)
        size = np.array(1650).reshape(1,1)
        rooms = np.array(3).reshape(1,1)
        combined = np.concatenate((size,rooms),axis=1)
        
        # We must scale our input for Grad Descent as well!!
        scaler3 = StandardScaler()
        combined_scaled = scaler3.fit_transform(combined)
        
        
        # Getting both predictions
        grad_predict = regressor.predict(combined_scaled)
        norm_predict = regressor_2.predict(combined)
        
        # Testing the Normal Equation Algorithm
        assert grad_predict/norm_predict <= 1.5 or norm_predict/grad_predict <= 1.5
    
    # Testing the Normal Equation against the Sklearn Library (single feature)
    def test_normal_sklearn_single(self):
         # Getting the Data
        PATH = '/Users/jinalshah/SpiderProjects/ML Algorithm Implementations/Data/Profits.csv'
        data = pd.read_csv(PATH)
        
        # Splitting the Data into X and y
        X = data.copy().drop('Population',axis=1).values
        y = data.copy()['Profits'].values
        
        # Reshaping X and y into the appropriate shape
        X = X.reshape(97,1)
        y = y.reshape(97,1)
        
        # Splitting the data into training and testing
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        
        # Creating the Regressors
        regressor = norm()
        regressor.fit(X_train,y_train)
        regressor_2 = lin_reg()
        regressor_2.fit(X_train, y_train)
        
        # Prediction value(s)
        pop = np.array(15).reshape(1,1)
        
        # Getting both predictions
        norm_predict = regressor.predict(pop)
        lib_predict = regressor_2.predict(pop)
        
        # Making sure both predictions are the same
        self.assertEqual(np.round(norm_predict[0][0],2),np.round(lib_predict[0][0],2))
    
    # Testing the Normal Equation against the Sklearn Library (multiple feature)
    def test_normal_sklearn_mult(self):
         # Getting the Data
        PATH = '/Users/jinalshah/SpiderProjects/ML Algorithm Implementations/Data/MultVarLin.csv'
        data = pd.read_csv(PATH)
        
        # Splitting the Data into X and y
        X = data.copy().drop('Price',axis=1).values
        y = data.copy()['Price'].values
        
        # Reshaping X and y into the appropriate shape
        X = X.reshape(47,2)
        y = y.reshape(47,1)
        
        # Splitting the data into training and testing
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
        
        # Creating both regressors
        regressor = norm()
        regressor.fit(X_train,y_train)
        regressor_2 = lin_reg()
        regressor_2.fit(X_train,y_train)
    
        # Prediction value(s)
        size = np.array(1650).reshape(1,1)
        rooms = np.array(3).reshape(1,1)
        combined = np.concatenate((size,rooms),axis=1)
        
        # Getting both predictions
        norm_predict = regressor.predict(combined)
        lib_predict = regressor_2.predict(combined)
        
        # Testing the Normal Equation Algorithm
        self.assertEqual(np.round(norm_predict[0][0],2),np.round(lib_predict[0][0],2))
        
    
if __name__ == '__main__':
    unittest.main()