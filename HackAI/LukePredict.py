import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import loadtxt

dataset = loadtxt('./cleanedDataNoHeader.csv', delimiter=',')
X = dataset[:,1:12]
y = dataset[:,12]
y = (y - 2)/7

print("Load saved model? (Y/N)   ")
response = input()
path = ""
if response == "Y":
    print("Which model?   ")
    generation = int(input())
    path = "./tensormodels/models/savedModel" + str(generation)
    model = keras.models.load_model(path)
else:
    model = Sequential()
    model.add(Dense(2, input_shape=(11,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    generation = int(loadtxt("./tensormodels/gen.txt"))+1
    path = "./tensormodels/models/savedModel" + str(generation)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# compile the keras model
model.compile(loss='mean_squared_error', optimizer=optimizer)

#epochs
N = 1000
#rolling average period
rap = 25

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose=1, start_from_epoch=300)

history = model.fit(X, y, epochs = N, batch_size=40, validation_split=0.3, shuffle=True, verbose=1, callbacks=[es_callback])

histories = history.history
history1 = histories['loss']
rollingAverage1 = [np.mean(history1[i:i+rap]) for i in range(len(history1)-rap)]
history2 = histories['val_loss']
rollingAverage2 = [np.mean(history2[i:i+rap]) for i in range(len(history2)-rap)]
epoch = list(range(0,len(history1)))
rollingEpochs = list(range(0,len(rollingAverage1)))

# df1 = pd.DataFrame(history1)
# havg1 = df1.rolling(window=7).mean()
# df2 = pd.DataFrame(history2)
# havg2 = df2.rolling(window=7).mean()

# plt.plot(epoch[25:], history1[25:], label='loss')
# plt.plot(epoch[25:], history2[25:], label='val_loss')
plt.plot(rollingEpochs[25:], rollingAverage1[25:], label='loss')
plt.plot(rollingEpochs[25:], rollingAverage2[25:], label='val_loss')
plt.legend()
model.save(path)

with open('./tensormodels/gen.txt', 'w') as f:
    f.write(str(generation))

# make probability predictions with the model
predictions = model.predict(X) * 5 + 3
# round predictions 
rounded = [round(x[0]) for x in predictions]
Error = []
correct = 0
for i in range(len(rounded)):
    Error.append(round(y[i] * 7 + 2 - rounded[i]))
    if (round(y[i]*7 + 2 - rounded[i])) == 0:
        correct += 1

print("Percentage correct guesses:   " + str(correct / len(Error)))
# plt.hist(Error)
# plt.hist(y*10)
#plt.hist(Error+y*10)
plt.show()

