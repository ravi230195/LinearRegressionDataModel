# LinearRegressionDataModel
LenearRegression Data Model for CO2 emission in Cars

Data Source 
  !wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv

Model statistics
  1.SimpleLinearRegressionDataModel
    Lenght of Training DataSet[853], Testing Dataset [214]

    Training and test Analysis for X-axis [ENGINESIZE] and Y-Axis [CO2EMISSIONS]
    coeficient [38.795124], intercept [127.169900]
    Mean absolute error: 20.60
    Residual sum of squares (MSE): 746.4
    R2-score: 0.71

    Training and test Analysis for X-axis [CYLINDERS] and Y-Axis [CO2EMISSIONS]
    coeficient [30.502243], intercept [79.987988]
    Mean absolute error: 25.01
    Residual sum of squares (MSE): 1104.71
    R2-score: 0.63

    Training and test Analysis for X-axis [FUELCONSUMPTION_COMB] and Y-Axis [CO2EMISSIONS]
    coeficient [15.621724], intercept [74.537788]
    Mean absolute error: 15.92
    Residual sum of squares (MSE): 461.98
    R2-score: 0.74
      ￼￼
  2.MulitpleLinearRegressionDataModel
    Lenght of Training DataSet[853], Testing Dataset [214]
    Training and test Analysis for X-axis [indepentedentData] and Y-Axis [CO2EMISSIONS]
    confient:  [[9.95870824 9.1406684  8.9030848 ]]
    intercept:  [66.35050925]
    Mean absolute error: 14.16
    Residual sum of squares (MSE): 363.62
    R2-score: 0.86
