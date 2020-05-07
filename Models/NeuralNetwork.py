import numpy as np
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from random import randint
from PIL import Image


class NeuralNetwork:
    def __init__(self):
        self._model = Sequential()
    
    def predict(self, X):
        return self._model.predict(X)
    
    def save(self, file_name):
        self._model.save_weights(file_name)
    
    #Allows to skip the training(fit) part 
    def load_weights(self, file_name):
        self._model.load_weights(file_name) 
    
    #Evaluate performances and if wanted print them
    def _evaluate(self, Xtest, ytest, display = True):
        test_loss_accuracy = self._model.evaluate(Xtest, ytest)
        if display:
            print("\nThe Loss Function on the Test set has a value of: {0:.1f}.".format(test_loss_accuracy[0]))
            print("The Accuracy on the Test set is: {0:.1f}%.\n".format(test_loss_accuracy[1]*100))
        return test_loss_accuracy
    
    #Display Original, Smoothed and De-Blurred image of a random element of the given set
    def _display_sample(self,X,y):
        i = randint(0,y.shape[0]-1)
        data = [X, y, self.predict(X).astype('uint8')]
        for batch in data:
            img = Image.fromarray(batch[i,:,:,:], 'RGB')
            img.show()