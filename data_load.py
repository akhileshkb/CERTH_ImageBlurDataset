import numpy as np
import pandas as pd
import os
import pickle
from keras.preprocessing import image
import tensorflow as tf

X_train = []
y_train = []
X_test = []
y_test = []


input_size = (300, 300)
folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Undistorted/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size = input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(0)
    else:
        print(filename, 'not a pic')
print("Trainset: Undistorted loaded...")

folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Artificially-Blurred/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(1)
    else:
        print(filename, 'not a pic')
print("Trainset: Artificially Blurred loaded...")

folderpath = 'CERTH_ImageBlurDataset/TrainingSet/Naturally-Blurred/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_train.append((1/255)*np.asarray(img))
        y_train.append(1)
    else:
        print(filename, 'not a pic')
print("Trainset: Naturally Blurred loaded...")

with open('X_train_300.pkl', 'wb') as picklefile:
    pickle.dump(X_train, picklefile)


with open('y_train_300.pkl', 'wb') as picklefile:
    pickle.dump(y_train, picklefile)


dgbset = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet.xlsx')
nbset = pd.read_excel('CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet.xlsx')

dgbset['MyDigital Blur'] = dgbset['MyDigital Blur'].apply(lambda x : x.strip())
dgbset = dgbset.rename(index=str, columns={"Unnamed: 1": "Blur Label"})

nbset['Image Name'] = nbset['Image Name'].apply(lambda x : x.strip())

folderpath = 'CERTH_ImageBlurDataset/EvaluationSet/DigitalBlurSet/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_test.append((1/255)*np.asarray(img))
        blur = dgbset[dgbset['MyDigital Blur'] == filename].iloc[0]['Blur Label']
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(filename, 'not a pic')
print("Testset: Artificially Blurred loaded...")

folderpath = 'CERTH_ImageBlurDataset/EvaluationSet/NaturalBlurSet/'

# load image arrays
for filename in os.listdir(folderpath):
    if filename != '.DS_Store':
        imagepath = folderpath + filename
        img = image.load_img(imagepath, target_size=input_size)
        X_test.append((1/255)*np.asarray(img))
        blur = nbset[nbset['Image Name'] == filename.split('.')[0]].iloc[0]['Blur Label']
        if blur == 1:
            y_test.append(1)
        else:
            y_test.append(0)
    else:
        print(filename, 'not a pic')

print("Trainset: Naturally Blurred loaded...")

with open('X_test_300.pkl', 'wb') as picklefile:
    pickle.dump(X_test, picklefile)


with open('y_test_300.pkl', 'wb') as picklefile:
    pickle.dump(y_test, picklefile)

