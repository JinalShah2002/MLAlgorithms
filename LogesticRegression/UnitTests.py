#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author Jinal Shah

This file will be used to run unit
tests for the Logestic Regression 
implementations


"""
# Importing the Libraries
import pandas as pd
import numpy as np
from LogesticRegression import LogesticRegression
import unittest
from RegLogesticRegression import LogesticRegression as reg

class TestLinearRegression(unittest.TestCase):
    
    # Testing the sigmoid function
    def test_sigmoid(self):
        
        # Regressor 
        regressor = LogesticRegression()
        
        self.assertEqual(regressor.sigmoid(0),0.5)
    
    # Testing sigmoid on matrices
    def test_sigmoid_matrices(self):
        # Creating a temp and answer
        temp = np.array(([0,0],[0,0],[0,0]))
        answer = np.array(([0.5,0.5],[0.5,0.5],[0.5,0.5]))
        
        # Regressor
        regressor = LogesticRegression()
        
        self.assertEqual(regressor.sigmoid(temp).all(),answer.all())
    
    # Testing the cost function
    def test_cost(self):
        PATH = '/Users/jinalshah/Jinal/Github Repos/MLAlgorithms/Data/CollegeAdmins.csv'
        raw_data = pd.read_csv(PATH)

        # Splitting into X and y
        X = raw_data.copy().drop('College Admission Decision',axis=1).values
        y = raw_data['College Admission Decision'].values
        
        # Building the Regressor
        regressor = LogesticRegression()
        regressor.fit(X,y)
        
        # Evaluating cost
        self.assertEqual(round(regressor.getCost(),3),0.693)
    
    # Testing Reg Cost
    def test_reg_cost(self):
        PATH = '/Users/jinalshah/Jinal/Github Repos/MLAlgorithms/Data/Microchips.csv'
        raw_data = pd.read_csv(PATH)

        # Splitting into X and y
        X = raw_data.copy().drop('Result',axis=1).values
        y = raw_data['Result'].values
        
        # Building the Regressor
        regressor = reg()
        regressor.fit(X,y)
        
        # Evaluating cost
        self.assertEqual(round(regressor.getCost(),2),0.7)
        


# Main Method
if __name__ == '__main__':
    unittest.main()

