import numpy as np 
from Models.Kernels import kernel_generator
from Models.Loader import Loader
from Models.Patcher import patcher
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d
from random import randint, sample, choice,  shuffle
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

    def _getData(self, data = "train"):
        switcher = {
            "train" :{
                "couples" : [(folder, image) for folder in self._train_folders for image in range(self._n_images)],
                "path" : self._train_path[1]
            },
            "validation" : {
                "couples" : [(folder, image) for folder in self._val_folders for image in range(self._n_images)],
                "path" : self._train_path[1]
            },
            "test" : {
                "couples" : [(folder, image) for folder in self._test_folders for image in range(self._n_images)],
                "path" : self._test_path[1]
            }
        }
        couples = switcher[data]["couples"]
        path = switcher[data]["path"]
        shuffle(couples)
        while True:
            for couple in couples:
                Xsharp = self._uncrypt(path,couple)
                X, y = self._motion_convolution(Xsharp, n_patches = 2190)
                np.random.shuffle(X)
                np.random.shuffle(y)
                yield (X, y)

    def get_Train_Test_Validation(self):
        return self._getData

    def _genCouples(self, folders, batch_size):
        couples = []
        while not(len(couples) == batch_size):
            tmp = (folders[randint(0, len(folders)-1)], randint(0, self._n_images - 1))
            if tmp in couples:
                continue
            else:
                couples.append(tmp)
        return couples

    def _motion_convolution(self, pic, dim = (30,32), stride = None, n_patches = None, filters = 1):
        if not n_patches:
            n_patches = int((pic.shape[0] * pic.shape[1]) /(dim[0] * dim[1]))    
        X = np.zeros((n_patches, dim[0], dim[1], 3))
        y = np.zeros((n_patches, 73))
        kernels = kernel_generator()
        index = [i for l in range(int(n_patches/len(kernels))) for i in range(len(kernels))]
        shuffle(index)
        patches = extract_patches_2d(pic, (30,32), max_patches = n_patches)
        for i in range(n_patches):
            X[i] =  filter2D(patches[i], -1, kernels[index[i]])
            y[i, index[i]] = 1
        """
        index = [randint(0, len(kernels) - 1) for i in range(n_patches)] 
        patches = patcher(pic, shape = dim, stride = stride,  n_patches = n_patches)
        for i in range(n_patches):
            X[i] =  filter2D(patches[i], -1, kernels[index[i]])
            y[i, index[i]] = 1
        """
        
        return X, y

    


    