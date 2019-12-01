from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

# seed
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df_pre = pd.read_csv('dino.csv', header=None)
df = df_pre.sample(frac=1)

dataset = df.values
X = dataset[:,0:6]
Y = dataset[:,6]


model = Sequential()
model.add(Dense(30,  input_dim=6, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
           optimizer='adam',
           metrics=['accuracy'])

model.fit(X, Y, epochs=1000, batch_size=200)

model.save('dino.h5')

print("\n Accuracy: %.4f" % (model.evaluate(X, Y)[1]))