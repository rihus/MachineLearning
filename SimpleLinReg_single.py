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
from sklearn.metrics import r2_score

##Load dataset
fuel_con = pd.read_csv("FuelConsumption.csv")

##To take a peek at dataset (default top 5 lines)
fuel_con.head()
##To detail few basic qualities of dataset
fuel_con.describe()
##to select few features to explore
selection_fcon=fuel_con[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY',
                         'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',
                         'FUELCONSUMPTION_COMB_MPG','CO2EMISSIONS']]
##To take a peek at selected dataset (top 10 lines)
selection_fcon.head(10)

##Plot all selected features as histograms
selection_fcon.hist()

##Plotting each feature vs the Emission (to see if their relation is linear)
##Engine Size
plt.figure(1)
plt.scatter(selection_fcon.ENGINESIZE, selection_fcon.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()
##Cylinders
plt.figure(2)
plt.scatter(selection_fcon.CYLINDERS, selection_fcon.CO2EMISSIONS, color='red')
plt.xlabel("Cylinders")
plt.ylabel("CO2 Emissions")
plt.show()
##Fuel Consumtion City
plt.figure(3)
plt.scatter(selection_fcon.FUELCONSUMPTION_CITY, selection_fcon.CO2EMISSIONS, color='green')
plt.xlabel("City Fuel Consumption")
plt.ylabel("CO2 Emissions")
plt.show()
##Fuel Consumtion Highway
plt.figure(4)
plt.scatter(selection_fcon.FUELCONSUMPTION_HWY, selection_fcon.CO2EMISSIONS, color='black')
plt.xlabel("Highway Fuel Consumption")
plt.ylabel("CO2 Emissions")
plt.show()
##Fuel Consumtion Combined
plt.figure(5)
plt.scatter(selection_fcon.FUELCONSUMPTION_COMB, selection_fcon.CO2EMISSIONS, color='cyan')
plt.xlabel("Combined Fuel Consumption")
plt.ylabel("CO2 Emissions")
plt.show()
##Fuel Consumtion Combined MPG
plt.figure(6)
plt.scatter(selection_fcon.FUELCONSUMPTION_COMB_MPG, selection_fcon.CO2EMISSIONS, color='magenta')
plt.xlabel("MPG Combined Fuel Consumption")
plt.ylabel("CO2 Emissions")
plt.show()
## Clearly, from the plots: MPG Combined Fuel Consumption in not linear

#############################################################
## For linear regression fit on combined fuel combustion:
##First we divide the data into train and test (~80 and 20% of the dataset, respectively)
rand_sel=np.random.rand(len(fuel_con)) < 0.8
train_fcon =selection_fcon[rand_sel]
test_fcon =selection_fcon[~rand_sel]
##Let's plot and see the training data
plt.figure(7)
plt.scatter(train_fcon.FUELCONSUMPTION_COMB, train_fcon.CO2EMISSIONS, color='cyan')
plt.xlabel("Combined Fuel Consumption Training Data")
plt.ylabel("CO2 Emissions")
plt.show()
##Modeling of linear regression
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train_fcon[['FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train_fcon[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
##Printing the coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)
##Plotting the fit line over data
plt.figure(8)
plt.scatter(train_fcon.FUELCONSUMPTION_COMB, train_fcon.CO2EMISSIONS, color='cyan')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Combined Fuel Consumption Training Data")
plt.ylabel("CO2 Emissions")

##TO claculate the accuracy of regression model above, we claculate the mean
##absolute error and mean squared error on the test data
test_x = np.asanyarray(test_fcon[['FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test_fcon[['CO2EMISSIONS']])
test_y_=regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
##R-squared is a metric for accuracy of model. It represents how close
##the data are to the fitted regression line. The higher the R-squared,
##the better the model fits the data. Best possible score is 1.0 
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )

#############################################################
## For linear regression fit on Cylinders:
# ##First we divide the data into train and test (~80 and 20% of the dataset, respectively)
# rand_sel=np.random.rand(len(fuel_con)) < 0.8
# train_fcon =selection_fcon[rand_sel]
# test_fcon =selection_fcon[~rand_sel]
##Let's plot and see the training data
# plt.scatter(train_fcon.CYLINDERS, train_fcon.CO2EMISSIONS, color='red')
# plt.xlabel("Cylinders Training Data")
# plt.ylabel("CO2 Emissions")
# plt.show()
##Modeling of linear regression
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train_fcon[['CYLINDERS']])
train_y = np.asanyarray(train_fcon[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
##Printing the coefficients
print('Coefficients_Cylinders: ', regr.coef_)
print('Intercept_Cylinders: ', regr.intercept_)
##Plotting the fit line over data
plt.figure(9)
plt.scatter(train_fcon.CYLINDERS, train_fcon.CO2EMISSIONS, color='red')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-b')
plt.xlabel("Cylinders Training Data")
plt.ylabel("CO2 Emissions")

##TO claculate the accuracy of regression model above, we claculate the mean
##absolute error and mean squared error on the test data
test_x = np.asanyarray(test_fcon[['CYLINDERS']])
test_y = np.asanyarray(test_fcon[['CO2EMISSIONS']])
test_y_=regr.predict(test_x)
print("Mean absolute error (Cylinders): %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE) - Cylinders: %.2f" % np.mean((test_y_ - test_y) ** 2))
##R-squared is a metric for accuracy of model. It represents how close
##the data are to the fitted regression line. The higher the R-squared,
##the better the model fits the data. Best possible score is 1.0 
print("R2-score (Cylinders): %.2f" % r2_score(test_y_ , test_y) )











