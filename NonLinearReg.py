# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 23:14:12 2021

@author: riazh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

##reading csv file with China's gdp data
china_gdp = pd.read_csv("china_gdp.csv")
##Let's plot and see the data
plt.figure(1) # figsize=(8,5) 
x_x, y_y = (china_gdp["Year"].values, china_gdp["Value"])
plt.plot(x_x, y_y, 'ro')
plt.ylabel("GDP")
plt.xlabel("Year")
plt.show
##Possible fit: Logistic function
def sigmoid_func(x, beta_1, beta_2):
    y = 1 / (1 + np.exp(-beta_1*(x-beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0
y_predict = sigmoid_func(x_x, beta_1, beta_2)
plt.plot(x_x, y_predict*1.5*10**13)
##Normalizing the values
xx=x_x/max(x_x)
yy=y_y/max(y_y)
##Fitting with non-linear least squares
popt, pcov = curve_fit(sigmoid_func, xx, yy)
print("beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))
##Plotting the data and fit
plt.figure(2)
x = np.linspace(1960, 2015, 55)
x = x/max(x)
y = sigmoid_func(x, *popt)
plt.plot(xx, yy, 'ro', label='data')
plt.plot(x, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()



