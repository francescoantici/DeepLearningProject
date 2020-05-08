import numpy as np 
from Models.Kernels import kernel_generator
from Models.Loader import Loader
from Models.Patcher import patcher
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
        #for index in range(len(images)):
        f = images[0]
        img = images[1]
        if f < 10:
            folder = "/00" + str(f)
        elif f < 100:
            folder = "/0" +str(f)
        else:
            folder = "/" + str(f)
        sys.stdout.flush()
        #print("\r"+files+":"+str(index)+"/"+str(len(images)), sep = ' ', end = '', flush = True)
        if img < 10:
            image = "/0000000" + str(img) + ".png"
        elif img < 100:
            image = "/000000" + str(img) + ".png"
        pic = np.asarray(Image.open(self._abs_path+"/"+files+folder+image), dtype = 'uint8')
        #out.append(pic)
        #sys.stdout.flush()
        #print("\r"+files+":    loaded    ", flush = True)
        #return np.asarray(out)
        return pic

    def _getTrain(self, batch_size):
        if batch_size:
            couples = self._genCouples(self._train_folders, batch_size)
        else:
            couples = [(folder, image) for folder in self._train_folders for image in range(self._n_images)]
        while True:
            index = 0
            for couple in couples:
                #print("\r"+"train"+":"+str(index)+"/"+str(len(couples)), sep = ' ', end = '', flush = True)
                Xsharp = self._uncrypt(self._train_path[0],couple)
                X, y = self._motion_convolution(Xsharp)
                index += 1
                #return X, y 
                yield (X, y)
       
        
    
    def _getVal(self, batch_size):
        if batch_size:
            couples = self._genCouples(self._val_folders, batch_size)
        else:
            couples = [(folder, image) for folder in self._val_folders for image in range(self._n_images)]
        while True:
            index = 0
            for couple in couples:
                #print("\r"+"validation"+":"+str(index)+"/"+str(len(couples)), sep = ' ', end = '', flush = True)
                Xsharp = self._uncrypt(self._train_path[0], couple)
                X, y = self._motion_convolution(Xsharp)
                index += 1
                #return X, y 
                yield (X, y)
                
    def _getTest(self, batch_size = None):
        if batch_size:
            couples = self._genCouples(self._test_folders, batch_size)
        else:
            couples = [(folder, image) for folder in self._test_folders for image in range(self._n_images)]
        while True:
            index = 0
            for couple in couples:
                #print("\r"+"test"+":"+str(index)+"/"+str(len(couples)), sep = ' ', end = '', flush = True)
                Xsharp = self._uncrypt(self._test_path[0], couple)
                X, y = self._motion_convolution(Xsharp)
                index += 1
                #return X, y 
                yield (X, y)

    def get_Train_Test_Validation(self):
        return (self._getTrain, self._getVal, self._getTest)

    def _genCouples(self, folders, batch_size):
        couples = []
        while not(len(couples) == batch_size):
            tmp = (folders[randint(0, len(folders)-1)], randint(0, self._n_images - 1))
            if tmp in couples:
                continue
            else:
                couples.append(tmp)
        return couples

    def _motion_convolution(self, pic, dim = (30,32)):
        n_patches = int((pic.shape[0] * pic.shape[1])/(dim[0] * dim[1]))
        #batch_size = Xsharp.shape[0] * n_patches
        X = np.zeros((n_patches, dim[0], dim[1], 3))
        y = np.zeros((n_patches, 73))
        chosen = []
        kernels = kernel_generator()
        count = 0
        #for pic in Xsharp:
        ymodel = np.zeros((73,))
        index = randint(0, len(kernels) - 1)
        while index in chosen:
            if len(chosen) >= len(kernels):
                break
            else:
                index = randint(0, len(kernels) - 1)
                continue

        chosen.append(index)
        patches = patcher(filter2D(pic, -1, kernels[index]), dim)
        ymodel[index] = 1
        for elem in range(n_patches):
            y[(count * n_patches) + elem] = ymodel
            X[(count * n_patches) + elem] = patches[elem]
        count += 1
        return np.asarray(X), np.asarray(y)

    


    