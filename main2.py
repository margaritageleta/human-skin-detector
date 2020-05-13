import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from SkinDetector import SkinDetector

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers import Flatten, BatchNormalization

def baseline_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(480,640,1)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    return model


def main():
    sd = SkinDetector()
    train = sd.segment_dataset(sd.TR_DATA)
    train_data = []
    for i in range(0,60):
        result = np.zeros((480,640))
        result[:train[i].shape[0],:train[i].shape[1]] = train[i]
        train_data.append(result)
    train_data = np.asarray(train_data)
    train_labels = np.asarray(sd.TR_LABEL)

    test = sd.segment_dataset(sd.VD_DATA)
    test_data = []
    for i in range(0,test.shape[0]):
    	result = np.zeros((480,640))
    	result[:test[i].shape[0],:test[i].shape[1]] = test[i]
    	test_data.append(result)

    test_data = np.asarray(test_data)
    test_labels = np.asarray(sd.VD_LABEL)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    rows, cols = 480, 640
    train_data = train_data.reshape(train_data.shape[0], rows, cols, 1)
    test_data = test_data.reshape(test_data.shape[0], rows, cols, 1)

    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')

    train_data /= 255.0
    test_data /= 255.0

    train_x, val_x, train_y, val_y = train_test_split(train_data, train_labels, test_size=0.2)

    batch_size = 256
    epochs = 5
    input_shape = (rows, cols, 1)

    model = baseline_model()
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    history = model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(val_x, val_y))

    predictions= model.predict(test_data)
    print(predictions)

main()
