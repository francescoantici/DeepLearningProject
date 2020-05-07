from Models.Loader import Loader
import numpy as np
from random import randint
from cv2 import GaussianBlur
from sklearn.model_selection import train_test_split

class Cifar10Loader(Loader):
    
    #Access just the training set
    def _getTrain(self):
        files = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
        xtrain = []
        ytrain = []
        d = self._uncrypt(files)
        for index in range(len(files)):
            n = d[files[index]][b'data'].shape[0]
            doc = files[index]
            for count in range(n):
                #Initialize a random Sigma value for the Gaussian Smoothing
                seed = randint(0,3)
                src = self.CreateImage(d[doc][b'data'][count,:])
                ytrain.append(src)
                dst = GaussianBlur(src, (5,5), seed)
                xtrain.append(dst) 
                

        return np.asarray(xtrain),np.asarray(ytrain)
   
    #Access just the Test set
    def _getTest(self):
        url = ["test_batch"]
        xtest = []
        ytest = []
        d = self._uncrypt(url)
        for index in range(len(url)):
            n = d[url[index]][b'data'].shape[0]
            doc = url[index]
            for count in range(n):
                #Initialize a random Sigma value for the Gaussian Smoothing
                seed = randint(0,3)
                src = self.CreateImage(d[doc][b'data'][count,:])
                ytest.append(src)
                dst = GaussianBlur(src, (5,5), seed)
                xtest.append(dst)
                

        return np.asarray(xtest),np.asarray(ytest)
    
    #Return the Data split into Train, Test and Validation sets (42 000, 9 000, 9 000)
    def get_Train_Test_Validation(self):
        url = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5","test_batch"]
        X = []
        y = []
        d = self._uncrypt(url)
        for index in range(len(url)):
            n = d[url[index]][b'data'].shape[0]
            doc = url[index]
            for count in range(n):
                #Initialize a random Sigma value for the Gaussian Smoothing
                seed = randint(0,3)
                src = self.CreateImage(d[doc][b'data'][count,:])
                y.append(src)
                dst = GaussianBlur(src, (5,5), seed)
                X.append(dst)
        #The proportion of the data is 70% Train, 15% Validation and 15% Test
        Xtrain, X_test, ytrain, y_test = train_test_split(np.asarray(X), np.asarray(y), test_size = 0.3)
        Xtest, Xval, ytest, yval = train_test_split(X_test, y_test, test_size = 0.5)
        return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

    #Parse every array of the Dataset into a Image-Like numpy array
    def CreateImage(self, vector):
        img = np.zeros((32,32,3),'uint8')
        img[:,:,0] = vector[0:1024].reshape((32,32))
        img[:,:,1] = vector[1024:2048].reshape((32,32))
        img[:,:,2] = vector[2048:].reshape((32,32))
        return img    

