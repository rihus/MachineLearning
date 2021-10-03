# -*- coding: utf-8 -*-
"""
Data and guide taken from Cognitive Class.ai
Copyright © 2018 [Cognitive Class](https://cocl.us/DX0108EN_CC).

Code written with Python 3.8 inside Spyder 3.2.5 (Anaconda 2.0.3)

Created on Sat Oct  2 12:56:00 2021

@author: riazh
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

##Load dataset
fuel_con = pd.read_csv("FuelConsumption.csv")

##select few features to explore
selection_fcon=fuel_con[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY',
                         'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',
                         'FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']]
##Plotting one feature vs the Emission (to see if their relation is linear)
##Engine Size
plt.figure(1)
plt.scatter(selection_fcon.ENGINESIZE, selection_fcon.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()
## For linear regression model fit on multiple independent datasets:
##First we divide the data into train and test (~80 and 20% of the dataset, respectively)
rand_sel=np.random.rand(len(fuel_con)) < 0.8
train_fcon =selection_fcon[rand_sel]
test_fcon =selection_fcon[~rand_sel]

##Modeling of multiple linear regression
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train_fcon[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train_fcon[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
##Printing the coefficients
print('Coefficients: ', regr.coef_)
print('Intercept ', regr.intercept_)

##TO claculate the accuracy of multiple regression model above, we claculate the
##ordinary least squares on the test data
y_hat = regr.predict(test_fcon[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test_fcon[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test_fcon[['CO2EMISSIONS']])

print("Residual sum of sqares: %.2f" % np.mean((y_hat - y)**2))
print("Variance score: %.2f" % regr.score(x,y))











