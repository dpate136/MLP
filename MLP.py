"""
Created on Thu Feb 14 13:13:09 2019
@author: dhara
"""

# -*- coding: utf-8 -*-

#Import Library 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import datetime

#import the dataset
dataset = pd.read_csv('filename.csv')

#adding variables
dataset["date"] = pd.to_datetime(dataset.date)
dataset["weekday"] = dataset.date.dt.dayofweek
dataset["month"] = dataset.date.dt.month
dataset["hour"] = dataset.date.dt.hour

#adding US Fedral Holidays
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start= dataset['date'].min(), end=dataset['date'].max()).to_pydatetime()
dataset['Holiday'] = dataset['date'].dt.date.astype('datetime64').isin(holidays)
dataset["Holiday"] = (dataset['Holiday'] == True).astype(int)
dataset.set_index('date', inplace = True)

#set the week, month and hour as a category
dataset.hour = dataset.hour.astype('category')
dataset.weekday = dataset.weekday.astype('category')
dataset.month = dataset.month.astype('category')

#splitting dataset into X(dependet variables) and Y(independent variable)
X = dataset.loc[:,dataset.columns != "Load"]
y = dataset["Load"]

#split the dataset into training and testing set
train_set = 0.70
idx = int((len(dataset) * train_set))
X_train = X.iloc[:idx, :]
X_test = X.iloc[idx:, :]
y_train = y.iloc[:idx]
y_test = y.iloc[idx:]

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = sc.fit_transform(y_train.values.reshape(-1, 1))
y_test = sc.transform(y_test.values.reshape(-1, 1))

def compute_mae(y, ypred):
    """given predicted and observed values, computes mean absolute error"""
    return np.mean(np.abs(ypred - y))

# train MLP model, and get validation set performance(ANN)
from sklearn.neural_network import MLPRegressor
def ann_mlp_mae(hl, a, lr, mi):
    model = MLPRegressor(hidden_layer_sizes=hl, alpha=a, learning_rate_init=lr, max_iter=mi)
    model.fit(X_train, y_train)
    pred_val = model.predict(X_test)
    return compute_mae(y_test, pred_val)

hidden_layer_sizes = [(10,),(20,),(15,30,),(40,60,),(20,30,40,)]
alpha = [0.0001, 0.00001, 0.001]
learning_rate = [0.0001, 0.001, 0.01]
max_iter = [200, 1000]
grid_search = pd.DataFrame(columns=['hl','a','lr','mi','mae'])

# perform grid search
for hl in hidden_layer_sizes:    
    for a in alpha:        
        for lr in learning_rate:
            for mi in max_iter:
                mae = get_mlp_mae(hl, a, lr, mi)
                params = {'hl':hl, 'a':a, 'lr':lr, 'mi':mi, 'mae':mae} 
                grid_search = grid_search.append(params, ignore_index=True)

# display best hyperparameters based on grid search
grid_search.sort_values('mae').head(1)

# best hyperparamters
hl, a, lr, mi  = (20,),  0.0001,  0.001,  200

# train model and get predictions
mod_mlp = MLPRegressor(hidden_layer_sizes=hl, alpha=a, learning_rate_init=lr, max_iter=mi)
mod_mlp.fit(X_train, y_train)
pred = mod_mlp.predict(X_test)

# compute error, and plot on both long and short time scales
print('MAE:', compute_mae(y_test, pred))

#mape
def mean_absolute_percentage_error(y_test, pred):
    MAPE = np.mean(np.abs((y_test - pred)/(y_test)))*100
    return MAPE

print('MAPE:',  mean_absolute_percentage_error(y_test, pred))


#graph for actual vs forecasted load
plt.plot(y_test, pred, lable = "actual vs forecast")

