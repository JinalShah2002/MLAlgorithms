#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author Jinal Shah

This will serve as my Neural Network Tester
Here, I am just testing that my Neural Network
can accurately calculate the XNOR logic gate

"""
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

# Getting the Data
PATH = '/Users/jinalshah/Jinal/Github Repos/MLAlgorithms/Data/XNOR.csv'
raw_data = pd.read_csv(PATH)

# Splitting data into X and Y
X = raw_data.copy().drop('Result',axis=1).values.reshape(4,2)
y = raw_data['Result'].values.reshape(4,1)

# Building the Predictor
neuralnet = NeuralNetwork()
neuralnet.fit(X,y)

