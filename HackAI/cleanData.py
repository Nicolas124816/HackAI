import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


trainData = pd.read_csv('./train.csv')

def getOutliers(trainData):
    filter = (trainData[["Id"]] >= 0)
    for i in trainData.columns[1:len(trainData.columns)-1]:
        # print("Column: " + i)
        q1 = np.percentile(trainData[i], 25)
        # print("q1: " + str(q1))
        q3 = np.percentile(trainData[i], 75)

        isNotOutlier1 = (trainData[[i]] < (q3 + 1.5 * (q3 - q1)))
        isNotOutlier2 = (trainData[[i]] > (q1 - 1.5 * (q3 - q1)))
        isNotOutlier1.columns=['Id']
        isNotOutlier2.columns=['Id']
        filter = (filter & isNotOutlier1) & isNotOutlier2
        # print(filter)
    
    return filter

filter = getOutliers(trainData)
# print("FILTER")
# print("---------------------------------------------------------------------------")
# print(newData)

# Removes all tuples with an outlier
def removeOutliers(trainData, filterData):
    # print("New data: ________________________")
    outliers = []
    for i in range(len(filterData["Id"])):
        if not(filterData["Id"][i]):
            outliers.append(i)

    trainData.drop(index = outliers, axis=0, inplace = True)

    return trainData


goodData = removeOutliers(trainData.copy(), filter)

# Generates a csv file from the data acter removing the outliers
goodData.to_csv('./cleanedData.csv', index=False)

# print(cleanData.shape)
# for i in cleanData.columns:
#     print("Hey")
#     cleanData.plot(kind = "scatter",x=i,y = "quality")
#     plt.show()

# for i in cleanData.columns:
#     cleanData.boxplot(column=i, by = "quality")
#     plt.show()

# print(cleanData.describe())
