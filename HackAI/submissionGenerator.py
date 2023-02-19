import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import loadtxt

print("Which model would you like to predict with?   ")
generation = int(input())
path = "./tensormodels/models/savedModel" + str(generation)
model = keras.models.load_model(path)

dataset = loadtxt('./testNoHeader.csv', delimiter=',')
X = dataset[:,1:12]

predictions = model.predict(X) * 7 + 2
rounded = [round(x[0]) for x in predictions]

savepath = './tensormodels/output/attempt' + str(generation) + '.csv' 

with open(savepath, 'w') as f:
    f.write("Id,quality")
    for i in range(len(rounded)):
        f.write("\n" + str(2056+i) + "," + str(rounded[i]))