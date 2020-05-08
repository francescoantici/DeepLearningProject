import numpy as np
from Models.NeuralNetwork import NeuralNetwork
from Models.Patcher import reconstruct
from NeuralNetworkReds.RedsLoader import RedsLoader
from keras.models import Sequential, Model
from keras.layers import Conv2D,Input, concatenate, MaxPooling2D, Dense, Activation, Flatten, Dropout, Reshape
from keras import backend as k
from keras.callbacks import EarlyStopping
from random import randint
from PIL import Image
import sys

class NeuralNetworkReds(NeuralNetwork):
    def __init__(self):
        self._model = Sequential()
        self._model.add(Conv2D(96, kernel_size = (7,7), activation = 'relu', input_shape = (30,32,3)))
        self._model.add(MaxPooling2D((2,2), strides = 2))
        self._model.add(Dropout(0.2))
        self._model.add(Conv2D(256, kernel_size = (5,5), activation = 'relu'))
        self._model.add(MaxPooling2D((2,2), strides = 2))
        self._model.add(Dropout(0.2))
        self._model.add(Flatten())
        self._model.add(Dense(1024, activation = 'relu'))
        self._model.add(Dropout(0.2))
        self._model.add(Dense(73))
        self._model.add(Activation('softmax'))
        self._model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
      

    def fit(self, arguments, epochs = 10):
        traingen, valgen, testgen = arguments
        es = EarlyStopping(monitor = 'val_accuracy', mode = max, restore_best_weights = True, verbose = 1)#, patience = 40)
        #es = EarlyStopping(monitor = 'val_losl', mode = min, restore_best_weights = True, verbose = 1, patience = 40)
        self._model.fit_generator(traingen(), steps_per_epoch = 21000, epochs = epochs, callbacks = [es], validation_data = valgen(), validation_steps = 3000)

            
            
    def evaluate(self, arguments, display = True):
        traingen, valgen, testgen = arguments
        return self._model.evaluate_generator(testgen(), steps = 3000)
    
    def display_sample(self, arguments):
        traingen, valgen, testgen = arguments
        X, y = testgen(1)
        data = [reconstruct(X, (720,1280)) , reconstruct(y, (720,1280)) , reconstruct(self.predict(X).astype('uint8'),(720,1280)) ]
        i = randint(0,data[0].shape[0]-1)
        for batch in data:
            img = Image.fromarray(batch[i], 'RGB')
            img.show()