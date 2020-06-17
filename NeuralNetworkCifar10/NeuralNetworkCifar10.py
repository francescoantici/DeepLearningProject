import numpy as np
from Models.Losses import MSE, PSNR, printer, ssim
from keras.models import Sequential, Model
from keras.layers import Conv2D, Input, concatenate
from keras.callbacks import EarlyStopping
from random import randint
from PIL import Image
from cv2 import GaussianBlur


class NeuralNetworkCifar10():
    def __init__(self):
        self.image_shape = (32,32,3)
        #The network is composed of 8 layers, and given the low dimensionality of the image the filter dimension is always pretty low
        """
        self._model = Sequential()
        image = Input(shape = self.image_shape)
        feat_extraction = Conv2D(filters = 64, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (32,32,3), use_bias = True) (image)
        feat_enhanced = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', use_bias = True) (feat_extraction)
        merge = concatenate([feat_extraction,feat_enhanced])
        second_order = Conv2D(filters = 64, kernel_size = (1,1), padding = 'valid' ,activation = 'relu', use_bias = True) (merge)
        third = Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', use_bias = True) (second_order)
        fourth = Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu', use_bias = True) (third)
        reconstruction = Conv2D(filters = 3, kernel_size = (3,3), padding = 'same', use_bias = True) (fourth)
        self._model = Model(inputs = image, outputs = reconstruction)
        self._model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy'])
        """
        self._model = Sequential()
        image = Input(shape = self.image_shape)
        feat_extraction = Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', activation = 'relu', input_shape = (32,32,3), use_bias = True) (image)
        feat_enhanced = Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu', use_bias = True) (feat_extraction)
        merge = concatenate([feat_extraction, feat_enhanced])
        second_order = Conv2D(filters = 32, kernel_size = (1,1), padding = 'same' ,activation = 'relu', use_bias = True) (merge)
        reconstruction = Conv2D(filters = 3, kernel_size = (3,3), padding = 'same', use_bias = True) (second_order)
        self._model = Model(inputs = image, outputs = reconstruction)
        self._model.compile(optimizer = 'adam', loss = 'mse', metrics=['accuracy'])
        
    def fit(self, arguments, epochs = 1):
        #The callback return the best set of Weights, given on the best value of the val_accuracy
        #After 3 non improving iteration it stops the training
        Xtrain, ytrain, Xval, yval, Xtest, ytest = arguments
        es = EarlyStopping(monitor = 'val_loss', mode = min, restore_best_weights = True, verbose = 1, patience = 30)
        return self._model.fit(Xtrain, ytrain, epochs = epochs, validation_data = (Xval, yval), callbacks = [es])

    def show_different_sigma(self, X, y, list_sigma):
        #Allows to show the effectiveness of the NeuralNetwork over different given values of Sigma 
        test = []
        index = randint(0, y.shape[0]-1)
        n = len(list_sigma)
        #Display the original image
        img = Image.fromarray(y[index,:,:,:], 'RGB')
        img.show(title = 'original')
        #Display the different smoothing of the same image
        for i in range(n):
            test.append(GaussianBlur(y[index,:,:,:],(5,5),list_sigma[i]))
            img1 = Image.fromarray(test[i], 'RGB')
            img1.show(title = 'smoothed sigma = ' + str(list_sigma[i]))
        #Display the deblurred images
        deblurred = self.predict(np.asarray(test)).astype('uint8')
        for i in range(n):
            img2 = Image.fromarray(deblurred[i,:,:,:], 'RGB')
            img2.show(title = 'reconstructed sigma = ' + str(list_sigma[i]))
        
    def evaluate(self, arguments, display = True):
        _, _, _, _, X, y = arguments
        y_pred = self._model.predict(X)
        SSIM = np.mean(np.asarray([ssim(y[i], y_pred[i]) for i in range(len(y))]))
        print("\nEvaluation over {} images from the test set\n".format(len(X)))
        printer(MSE(y, y_pred), SSIM, PSNR(y, y_pred))
        i = randint(0, y.shape[0] - 1)
        self.display_sample(X[i], y[i], y_pred[i])
        print("\nEvaluation on the displayed image\n")
        printer(MSE(y[i], y_pred[i]), ssim(y[i], y_pred[i]), PSNR(y[i], y_pred[i]))

    
    def save(self, file_name):
        self._model.save_weights(file_name)
    
    #Allows to skip the training(fit) part 
    def load_weights(self, file_name):
        self._model.load_weights(file_name) 
    
    #Display Original, Smoothed and De-Blurred image of a random element of the given set
    def display_sample(self, X, y, y_pred):
        img = np.concatenate((X, y_pred), axis = 1)
        data = [y, img.astype('uint8')]
        for batch in data:
            img = Image.fromarray(batch, 'RGB')
            img.show()
