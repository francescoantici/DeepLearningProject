import numpy as np 
from Models.Kernels import kernel_generator
from Models.Loader import Loader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d
from random import randint, sample, choice
import sys
from cv2 import filter2D
   
class RedsLoader(Loader):
    
    def __init__(self, path = "/home/francescoantici/dataset/RedsDataset"):
        #Useful paramters and paths
        
        self._abs_path = path 
        self._train_batch_size = 128
        self._val_batch_size = 32
        self._train_path = ["train_blur", "train_sharp"]
        self._test_path = ["val_blur", "val_sharp"]
        self._train_files = 240
        self._test_files = 30
        self._val_files = 30
        self._val_folders = sample(range(self._train_files), self._val_files)
        self._train_folders = [folder for folder in range(self._train_files) if not(folder in self._val_folders)]
        self._test_folders = [folder for folder in range(self._test_files)]
        self._n_images = 100

    def _uncrypt(self, files, images):
        out = []
        for index in range(len(images)):
            f = images[index][0]
            img = images[index][1]
            if f < 10:
                folder = "/00" + str(f)
            elif f < 100:
                folder = "/0" +str(f)
            else:
                folder = "/" + str(f)
            sys.stdout.flush()
            print("\r"+files+":"+str(index)+"/"+str(len(images)), sep = ' ', end = '', flush = True)
            if img < 10:
                image = "/0000000" + str(img) + ".png"
            elif img < 100:
                image = "/000000" + str(img) + ".png"
            pic = np.asarray(Image.open(self._abs_path+"/"+files+folder+image), dtype = 'uint8')
            out.append(pic)
        sys.stdout.flush()
        print("\r"+files+":    loaded    ", flush = True)
        return np.asarray(out)

    def _getTrain(self):
        couples = self._genCouples(self._train_folders, self._train_batch_size)
        ###Deconvolution method
        #Xsharp = self._uncrypt(self._train_path[0],couples)
        #X, y = self._motion_convolution(Xsharp)
        #return X, y 
        ###My Method
        X = self._uncrypt(self._train_path[0],couples)
        y = self._uncrypt(self._train_path[1],couples)
        return self._patches_creator(X, (30,32)), self._patches_creator(y, (30,32))
        
    
    def _getVal(self):
        couples = self._genCouples(self._val_folders, self._val_batch_size)
        ###Deconvolution Method
        #Xsharp = self._uncrypt(self._train_path[0], couples)
        #X, y = self._motion_convolution(Xsharp)
        #return X, y
        ###My method
        X = self._uncrypt(self._train_path[0],couples)
        y = self._uncrypt(self._train_path[1],couples)
        return self._patches_creator(X, (30,32)), self._patches_creator(y, (30,32))
        
    def _getTest(self, batch_size = None):
        if batch_size:
            couples = self._genCouples(self._test_folders, batch_size)
        else:
            couples = [(folder, image) for folder in self._test_folders for image in range(self._n_images)]
        ###Deconvolution Method
        #Xsharp = self._uncrypt(self._test_path[0], couples)
        #X, y = self._motion_convolution(Xsharp)
        #return X, y
        ###My method
        X = self._uncrypt(self._test_path[0],couples)
        y = self._uncrypt(self._test_path[1],couples)
        return self._patches_creator(X, (30,32)), self._patches_creator(y, (30,32))

    def get_Train_Test_Validation(self):
        return (self._getTrain, self._getVal, self._getTest)

    def _genCouples(self, folders, batch_size):
        couples = []
        while not(len(couples) == batch_size):
            tmp = (folders[randint(0, len(folders)-1)], randint(0, self._n_images))
            if tmp in couples:
                continue
            else:
                couples.append(tmp)
        return couples

    def _motion_convolution(self, Xsharp):
        X = []
        y = []
        chosen = []
        kernels = kernel_generator()
        for pic in Xsharp:
            ymodel = np.zeros((73,))
            index = randint(0, len(kernels) - 1)
            while index in chosen:
                if len(chosen) >= len(kernels):
                    break
                else:
                    index = randint(0, len(kernels) - 1)
                    continue

            chosen.append(index)
            patches = extract_patches_2d(filter2D(pic, -1, kernels[index]), (30,30))
            X.append(patches)
            ymodel[index] = 1
            for elem in range(patches.shape[0]):
                y.append(ymodel)
        return np.asarray(X), np.asarray(y)

    def _patches_creator(self, images, shape):
        out = []
        for i in range(images.shape[0]):
            for row in range(0, images[i].shape[0], shape[0]):
                for column in range(0, images[i].shape[1], shape[1]):
                    out.append(images[i, row:row+shape[0], column:column+shape[1], :])
        return np.asarray(out).reshape((len(out), shape[0], shape[1], 3))


    