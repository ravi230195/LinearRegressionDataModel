'''
 Data from !wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


def scaterPlot(X, Y, X_Label="", Y_LABEL = ""):
    '# scattered plot'
    plt.scatter(X, Y,  color='blue')
    plt.xlabel(X_Label)
    plt.ylabel(Y_LABEL)
    plt.show()
    
def trainingModel(X, Y):
    train_x = np.asanyarray(X)
    train_y = np.asanyarray(Y)
    reg = linear_model.LinearRegression()
    reg.fit(train_x, train_y)
    print ("coeficient [%f], intercept [%f]" %(reg.coef_, reg.intercept_))
    plt.scatter(X, Y,  color='blue')
    plt.plot(train_x, reg.coef_[0][0]*train_x + reg.intercept_[0], '-r')
    '''plt.xlabel(X)
    plt.ylabel(Y)'''
    plt.show()
    return reg

def testAndEvaluation(reg, X, Y):
    test_x = np.asanyarray(X)
    test_y = np.asanyarray(Y)
    test_y_hat = reg.predict(test_x)
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )

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

'# Picking Training Set 80% of Dataset'
index = (int)(len(cdf)*0.8)
train = cdf[:index]

'# Picking Test Set'
test = cdf[index:]
print ("Lenght of Training DataSet[%d], Testing Dataset [%d]" % (len(train), len(test)))


independentDataSet = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]
for i in independentDataSet:
    print("\nTraining and test Analysis for X-axis [%s] and Y-Axis [%s]" %(i,"CO2EMISSIONS"))
    reg = trainingModel(train[[i]], train[["CO2EMISSIONS"]])
    testAndEvaluation(reg, test[[i]] , test[["CO2EMISSIONS"]])
