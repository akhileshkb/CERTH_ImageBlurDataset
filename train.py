import numpy as np
import pandas as pd
import os
import pickle
from keras.preprocessing import image
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score

input_size = (224, 224)

with open('X_train.pkl', 'rb') as picklefile:
    X_train = pickle.load( picklefile)


with open('y_train.pkl', 'rb') as picklefile:
    y_train = pickle.load( picklefile)


with open('X_test.pkl', 'rb') as picklefile:
    X_test = pickle.load(picklefile)


with open('y_test.pkl', 'rb') as picklefile:
    y_test = pickle.load(picklefile)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(input_size[0], input_size[1], 3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (5,5), activation = 'relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(128, (5,5), activation = 'relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  # tf.keras.layers.Dense(256, activation='relu'),
  # tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

traindata = np.stack(X_train)
testdata = np.stack(X_test)
trainlabel = to_categorical(y_train)
testlabel = to_categorical(y_test)

# epochs = 10
# for i in range(epochs):
model.fit(traindata, trainlabel, batch_size=16, epochs=10, validation_data=(testdata, testlabel), verbose=1)
print("Model training complete...")
(loss, accuracy) = model.evaluate(testdata, testlabel, batch_size = 32, verbose = 1)
print("accuracy: {:.2f}%".format(accuracy * 100))

