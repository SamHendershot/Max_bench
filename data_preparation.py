#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 12:25:32 2022

@author: samuel
"""

import pandas as pd
import numpy as np

# -------------- Loading data/preparation  -------------- #

X_train = pd.read_csv("/Users/samuel/Desktop/Powerlifting/X_train.csv")
X_test = pd.read_csv("/Users/samuel/Desktop/Powerlifting/X_test.csv")
y_train = pd.read_csv("/Users/samuel/Desktop/Powerlifting/y_train.csv")
y_test = pd.read_csv("/Users/samuel/Desktop/Powerlifting/y_test.csv")


# Prepare data for Exploratory Data Analysis
def prepareData(X_train, X_test, y_train, y_test):
    X = pd.concat([X_train, X_test]).copy()
    y = pd.concat([y_train, y_test]).copy()
    
    X.drop('playerId', axis=1, inplace=True) # Dropping Id
    y.drop(['playerId', 'Age', 'BodyweightKg', 'BestDeadliftKg'], axis=1, inplace=True) # Age, bodyweight, and deadlift all 100% null, remove columns entirely
    
    fullData = pd.concat([X, y], axis=1)
    
    fullData['BestSquatKg'] = pd.to_numeric(fullData['BestSquatKg'], errors='coerce') # Transform squat column to float
    
    fullData.isnull().sum() / fullData.shape[0] * 100 # Check percentage of null values
    
    fullData = fullData.dropna().copy() # Since only 1.4% of 30,000 is null, we can drop the rows
    
    # Remove negative values
    
    for i in fullData.columns:
        if fullData.dtypes[i] == 'float64':
            fullData = fullData[fullData[i] > 0]
            
    # Separating male and female lifters
    
    datM = fullData[ fullData['Sex'] == 'M' ]
    datF = fullData[ fullData['Sex'] == 'F' ]

    return datM, datF, fullData

datM, datF, fullData = prepareData(X_train, X_test, y_train, y_test)

datM.to_csv('lifting_data_male.csv')
datF.to_csv('lifting_data_female.csv')
fullData.to_csv('whole_data_set.csv')
