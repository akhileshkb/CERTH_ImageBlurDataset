import numpy as np
import pandas as pd
import os
import pickle
from keras.preprocessing import image
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score

model_reload = tf.keras.models.load_model('model_keras_87')

print(model_reload.summary())

input_size = (224, 224)

with open('X_test.pkl', 'rb') as picklefile:
    X_test = pickle.load(picklefile)

with open('y_test.pkl', 'rb') as picklefile:
    y_test = pickle.load(picklefile)

testdata = np.stack(X_test)
testlabel = to_categorical(y_test)

(loss, accuracy) = model_reload.evaluate(testdata, testlabel, batch_size = 128, verbose = 1)
print("accuracy: {:.2f}%".format(accuracy * 100))