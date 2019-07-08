# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:41:07 2019

@author: erasunn
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from sklearn.metrics import r2_score

def scatterPlot(X, Y, X_label = "", Y_label=""):
    plt.scatter(X, Y, color = 'blue')
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.show()
    
def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y
 
def trainingModel(train_x, train_y, xdata, ydata):
    popt, pcov = curve_fit(sigmoid, train_x, train_y)
    print("parameters ", popt)
    print("convinance ", pcov)
    # plot the pridiction line
    plt.scatter(xdata, ydata, color = 'blue')
    plt.xlabel("year")
    plt.ylabel("value")
    XX = np.arange(1960,2010,1)
    XX_N = XX/max(XX)
    YY = sigmoid(XX_N, *popt)
    YY_N = YY/max(YY)
    plt.scatter(XX_N,YY_N, color = 'red')
    plt.show()

def testAndEvaluation(test_x, test_y, *popt):
    y_hat = sigmoid(test_x, *popt)
    # evaluation
    print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(y_hat , test_y))
 

############# MAIN##################
df = pd.read_csv(r'C:\Users\ERASUNN\Downloads\china_gdp.csv')
#print(df.head())
print(df.describe())
scatterPlot(df['Year'], df['Value'], "Year", "value")

#Scatter plot is a Non-lenear, best fit is parabolic curve
# y = c+ mX*2

x_data = df["Year"]
y_data = df["Value"]
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

print ("Lenght of Training DataSet[%d], Testing Dataset [%d]" % (len(train_x), len(test_x)))
trainingModel(train_x, train_y, xdata, ydata)
testAndEvaluation(test_x, test_y, *popt)

