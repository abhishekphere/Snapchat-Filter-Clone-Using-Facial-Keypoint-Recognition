"""
This is a sample program that implements Facial Keypoint recognition.
"""
import torch
import tensorflow as tf
import csv
import numpy as np

from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset():
    '''
    Load training dataset
    '''
    Xtrain = []
    Ytrain = []
    with open('training.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float)
            for i, val in enumerate(row["Image"].split(" ")):
                img[i // IMAGE_WIDTH, i % IMAGE_WIDTH, 0] = val
            Yitem = []
            failed = False
            for coord in row:
                if coord == "Image":
                    continue
                if (row[coord].strip() == ""):
                    failed = True
                    break
                Yitem.append(float(row[coord]))
            if not failed:
                Xtrain.append(img)
                Ytrain.append(Yitem)

    return np.array(Xtrain), np.array(Ytrain, dtype=np.float)


Xdata, Ydata = load_dataset()
Xtrain = Xdata[:]
Ytrain = Ydata[:]

def show_image(X, Y):
    img = np.copy(X)
    for i in range(0,Y.shape[0],2):
        if 0 < Y[i+1] < IMAGE_HEIGHT and 0 < Y[i] < IMAGE_WIDTH:
            img[int(Y[i+1]),int(Y[i]),0] = 255
    plt.imshow(img[:,:,0])

# Preview dataset samples
show_image(Xtrain[0], Ytrain[0])

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH,1) ))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(30))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mse',
              metrics=['mae'])

model.fit(Xtrain, Ytrain, epochs=1000)

def load_testset():
    Xtest = []
    with open('test.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float)
            for i, val in enumerate(row["Image"].split(" ")):
                img[i // IMAGE_WIDTH, i % IMAGE_WIDTH, 0] = val
            Xtest.append(img)

    return np.array(Xtest)

Xtest = load_testset()

def show_results(image_index):
    Ypred = model.predict(Xtest[image_index:(image_index+1)])
    show_image(Xtest[image_index], Ypred[0])
