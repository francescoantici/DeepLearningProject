import numpy as np 
from Models.Loader import Loader
from Models.Processing import preprocess
from PIL import Image
from random import randint, sample,  shuffle
   
class RedsLoader(Loader):
    
    def __init__(self, path = "/home/francescoantici/dataset/RedsDataset"):
        #Useful paramters and paths
        
        self._abs_path = path 
        self._train_files = 240
        self._test_files = 30
        self._val_files = 30
        self._val_folders = sample(range(self._train_files), self._val_files)
        self._train_folders = [folder for folder in range(self._train_files) if not(folder in self._val_folders)]
        self._test_folders = [folder for folder in range(self._test_files)]
        self._n_images = 100

    def _uncrypt(self, files, images):
        #Read a single image in the format (folder, image)
        out = []
        f = images[0]
        img = images[1]
        if f < 10:
            folder = "/00" + str(f)
        elif f < 100:
            folder = "/0" +str(f)
        else:
            folder = "/" + str(f)
        if img < 10:
            image = "/0000000" + str(img) + ".png"
        elif img < 100:
            image = "/000000" + str(img) + ".png"
        pic = Image.open(self._abs_path+"/"+files+folder+image)
        return pic

    def _getData(self, data = "train"):
        #Return a generator of the train/val/test set in the format (X, y)
        switcher = {
            "train" :{
                "couples" : [(folder, image) for folder in self._train_folders for image in range(self._n_images)],
                "path" : ["train_blur", "train_sharp"]
            },
            "validation" : {
                "couples" : [(folder, image) for folder in self._val_folders for image in range(self._n_images)],
                "path" : ["train_blur", "train_sharp"]
            },
            "test" : {
                "couples" : [(folder, image) for folder in self._test_folders for image in range(self._n_images)],
                "path" : ["val_blur", "val_sharp"]
            }
        }
        couples = switcher[data]["couples"]
        path = switcher[data]["path"]
        shuffle(couples)
        while True:
            for couple in couples:
                X = self._uncrypt(path[0],couple)
                y = self._uncrypt(path[1],couple)
                yield (X, y)

    def get_Train_Test_Validation(self):
        return self._getData("train"), self._getData("validation"), self._getData("test")

    

    


    