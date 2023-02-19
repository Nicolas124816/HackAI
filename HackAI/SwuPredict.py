
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

# Reads in data from already made files
goodData = pd.read_csv('./cleanedData.csv')
testData = pd.read_csv('./test.csv')

# Predicts data
def predict(train,test):
    x = train[['fixed acidity','volatile acidity','citric acid',
    'residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
    y = train[['quality']]
    regr = linear_model.LinearRegression()
    regr.fit(x.values , y.values)
    input = [1]
    predictedQualities=[]
    for i in range(len(test["Id"])):
        input[0] = [test.iloc[i]][0][1:12]
        predictedQualities.append(regr.predict(input))
    return predictedQualities

# Out holds the predicted data
out = predict(goodData,testData)

# Stores formatted output 
formattedOut = []
for i in range(len(out)):
    formattedOut.append(round(out[i][0][0]))

# Loads predicted data to a file
with open('./SwuPredict/submission25.csv', 'w') as f:
    f.write("Id,quality")
    for i in range(len(testData["Id"])):
        f.write("\n" + str(2056+i) + ", " + str(formattedOut[i]))

