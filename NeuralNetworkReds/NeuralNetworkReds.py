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
        """
        self._model = Sequential()
        self._model.add(Conv2D(96, kernel_size = (7,7), activation = 'relu', input_shape = (30,30,3)))
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
        """
        self._model = Sequential()
        image = Input(shape = (30,32,3))
        feat_extraction = Conv2D(filters = 64, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (30,32,3), use_bias = True) (image)
        feat_enhanced = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', use_bias = True) (feat_extraction)
        merge = concatenate([feat_extraction,feat_enhanced])
        second_order = Conv2D(filters = 64, kernel_size = (1,1), padding = 'valid' ,activation = 'relu', use_bias = True) (merge)
        third = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', use_bias = True) (second_order)
        fourth = Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu', use_bias = True) (third)
        reconstruction = Conv2D(filters = 3, kernel_size = (3,3), padding = 'same', use_bias = True) (fourth)
        self._model = Model(inputs = image, outputs = reconstruction)
        self._model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy'])

    def fit(self, arguments, epochs = 10):
        traingen, valgen, testgen = arguments
        #es = EarlyStopping(monitor = 'val_accuracy', mode = max, restore_best_weights = True, verbose = 1, patience = 40)
        es = EarlyStopping(monitor = 'val_loss', mode = min, restore_best_weights = True, verbose = 1, patience = 40)
        for epoch in range(epochs):
            sys.stdout.flush()
            print("\r"+str(epoch+1)+"/"+str(epochs), sep = ' ', end = '', flush = True)
            Xtrain, ytrain = traingen()
            Xval, yval = valgen()
            self._model.fit(Xtrain, ytrain, validation_data = (Xval, yval), callbacks = [es])
    
    
            
            
    def evaluate(self, arguments, display = True):
        trainge, valgen, testgen = arguments
        Xtest, ytest = testgen(16)
        return super()._evaluate(Xtest, ytest, display = display)
    
    def display_sample(self, arguments):
        trainge, valgen, testgen = arguments
        X, y = testgen(1)
        data = [reconstruct(X, (720,1280)) , reconstruct(y, (720,1280)) , reconstruct(self.predict(X).astype('uint8'),(720,1280)) ]
        i = randint(0,data[0].shape[0]-1)
        for batch in data:
            img = Image.fromarray(batch[i], 'RGB')
            img.show()