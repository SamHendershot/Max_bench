#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 11:16:10 2022

@author: samuel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import load_model

from sklearn.metrics import r2_score, mean_absolute_error

dat = pd.read_csv("/Users/samuel/Desktop/Powerlifting/whole_data_set.csv", index_col=0)

# -------------- Data prep -------------- #
 
def prepareData(dat):
    
    datM = dat[ dat['Sex'] == 'M' ]
    datF = dat[ dat['Sex'] == 'F' ]
    
    X = dat.drop(['BestBenchKg', 'BestDeadliftKg', 'BestSquatKg', 'Name'], axis=1)
    y = dat['BestBenchKg']
    
    # Binary Encoding
    X['Sex'] = X['Sex'].apply(lambda x: 1 if x == 'M' else 0)
    
    # One-hot encoding
    X = pd.concat([X, pd.get_dummies( X['Equipment'] )], axis=1)
    X.drop('Equipment', axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=1)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = prepareData(dat)

# -------------- Neural Network  -------------- #

class neuralNetwork:
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    def train(self):
        
        input_shape = X_train.shape[1]
        
        model = keras.Sequential([
            
            layers.BatchNormalization(),
            
            layers.Dense(512, activation='relu', input_shape=[input_shape]),
            layers.Dropout(.3),
            layers.BatchNormalization(),
            
            layers.Dense(512, activation='relu'),
            layers.Dropout(.3),
            layers.BatchNormalization(),
            
            layers.Dense(512, activation='relu'),
            layers.Dropout(.3),
            layers.BatchNormalization(),
            
            layers.Dense(1),
            
            ])
        
        model.compile(
            optimizer='adam',
            loss='mae',
            )
        
        early_stopping = callbacks.EarlyStopping(
            min_delta=.001,
            patience=20,
            restore_best_weights=True,
            )
        
        hist = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            batch_size=128,
            epochs=100,
            callbacks=[early_stopping],
            )
        
        model.save('network.h5')
        network = load_model('network.h5')
        
        return network, hist
    
    def loss(self, hist):
        hist_df = pd.DataFrame(hist.history)
        hist_df.loc[:, ['loss', 'val_loss']].plot()
        print('Min validation loss: {}'.format(hist_df['val_loss'].min()) )

network, hist = neuralNetwork(X_train, X_test, y_train, y_test).train()
neuralNetwork(X_train, X_test, y_train, y_test).loss(hist)


def predict(sex, age, weight, multi, raw, single, wrap):
    print("Your predicted max bench is {:.0f} lbs".format(
    network.predict(np.array([sex, age, weight, multi, raw, single, wrap]).reshape(1,-1))[0][0] * 2.205)
        )

predict(1, 22, 66, 0, 1, 0, 0)


