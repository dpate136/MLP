"""
Created on Thu Feb 14 13:13:09 2019
@author: dhara
"""

# -*- coding: utf-8 -*-

#Import Library 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('filename.csv')

#set the week, month and hour as a category
dataset.hour = dataset.hour.astype('category')
dataset.weekday = dataset.weekday.astype('category')
dataset.month = dataset.month.astype('category')

#splitting dataset into X and Y
X = dataset.iloc[:,1:17]
Y = dataset.iloc[:, 17]

#normlize the dataset
scaler = MinMaxScaler(feature_range=(0,1))
scaled_X = scaler.fit_transform(X)
scaled_y = scaler.fit_transform(Y)

#split the dataset into training and testing set(normlized data)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X, scaled_y, test_size = 0.25, random_state = 0)

#ANN
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(32,24,40))

mlp.fit(X_train,y_train)

y_pred = mlp.predict(X_test).reshape(-1,1)

inv_pred = scaler.inverse_transform(y_pred)
inv_test = scaler.inverse_transform(y_test)

from sklearn.metrics import mean_squared_error
from math import sqrt
rms1 = sqrt(mean_squared_error(inv_pred, inv_test))

#mape
def mean_absolute_percentage_error(inv_pred, inv_test): 
    inv_pred, inv_test = np.array(inv_test), np.array(inv_pred)
mape1 = np.mean(np.abs((inv_pred, inv_test) / inv_test)) * 100

plt.plot(inv_test, inv_pred, lable = "actual vs forecast")

