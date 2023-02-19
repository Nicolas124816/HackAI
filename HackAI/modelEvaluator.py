import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import loadtxt
    print("Which model would you like to evaluate?   ")
    generation = int(input())
    path = "./tensormodels/savedModel" + str(generation)

    dataset = loadtxt('./cleanedDataNoHeader.csv', delimiter=',')
    X = dataset[:,1:12]
    # the correct values for each thing
    y = dataset[:,12]
    model = keras.models.load_model(path)

    predictions = model.predict(X)
    # rounded is the final predictions for each value
    rounded = [round(x[0]*5 + 3) for x in predictions]

    def makeArray(N):
        array = [i in range(N)]

    #-----------------------------------------------------------------------------------
    # Dimension of the array
    N = 10

    # O Matrix and actual and predicted vectors
    O = np.zeros((N,N))
    actual = np.zeros(N)
    predicted = np.zeros(N)
    for n in range(len(y)):
        i = int(y[n])
        j = int(rounded[n])
        O[i][j] += 1
        actual[i] += 1
        predicted[j] += 1

    # Outer Product (E)
    E = np.outer(actual, predicted)
    E = E / np.sum(E) * len(y)

    # Weight Matrix
    W = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            W[i][j] = ((i-j)**2)/((N-1)**2)

    numerator = 0
    denominator = 0
    for i in range(N):
        for j in range(N):
            numerator +=    W[i][j] * O[i][j]
            denominator +=  W[i][j] * E[i][j]

    k = 1 - numerator/denominator

    print("The score is:     " + str(k))






