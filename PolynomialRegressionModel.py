'''
 Data from !wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def scaterPlot(X, Y, X_Label="", Y_LABEL = ""):
    '# scattered plot'
    plt.scatter(X, Y,  color='blue')
    plt.xlabel(X_Label)
    plt.ylabel(Y_LABEL)
    plt.show()
    
def trainingModel(X, Y):
    train_x = np.asanyarray(X)
    train_y = np.asanyarray(Y)
    #print (train_x)
    poly = PolynomialFeatures(degree=2)
    train_x_poly = poly.fit_transform(train_x)
    #print(train_x_poly)
    reg = linear_model.LinearRegression()
    reg.fit(train_x_poly, train_y)
    print ("coeficient ", reg.coef_)
    print ("Intercept " , (reg.intercept_))
    plt.scatter(X, Y,  color='blue')
    XX = np.arange(0,10,0.1)
    YY = reg.intercept_[0] + reg.coef_[0][1]*XX + reg.coef_[0][2]*np.power(XX,2)
    plt.plot(XX, YY, '-r')
    plt.xlabel("ENGINESIZE")
    plt.ylabel("EMISSION")
    plt.show()
    return reg

def testAndEvaluation(reg, X, Y):
    test_x = np.asanyarray(X)
    test_y = np.asanyarray(Y)
    poly = PolynomialFeatures(degree=2)
    test_x_poly = poly.fit_transform(test_x)
    #print(train_x_poly)
    test_y_hat = reg.predict(test_x_poly)
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
    print('Variance score: %.2f' % reg.score(test_x_poly, test_y))

########  MAIN  ##################
df = pd.read_csv(r'C:\Users\ERASUNN\Downloads\FuelConsumptionCo2.csv')

'#summarize the data'
summarize_data = df.describe()
'#exract required data'
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#print (cdf.describe())
#print (summarize_data)

'#Histogram graph for CDF dats set'
cdf.hist()
plt.show()

'#Scater Plot for the independentData'
independentDataSet = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]
for i in independentDataSet:
    scaterPlot(cdf[i], cdf.CO2EMISSIONS, i, "CO2EMISSIONS")
    
print("We Can See That the ENGINESIZE scatter plot is slighly curved")

'# Picking Training Set 80% of Dataset'
index = (int)(len(cdf)*0.8)
train = cdf[:index]

'# Picking Test Set'
test = cdf[index:]
print ("Lenght of Training DataSet[%d], Testing Dataset [%d]" % (len(train), len(test)))


independentDataSet = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]
print("\nTraining and test Analysis for X-axis [%s] and Y-Axis [%s]" %(independentDataSet[0],"CO2EMISSIONS"))
reg = trainingModel(train[[independentDataSet[0]]], train[["CO2EMISSIONS"]])
testAndEvaluation(reg, test[[independentDataSet[0]]] , test[["CO2EMISSIONS"]])
