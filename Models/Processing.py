import numpy as np
from PIL import Image


def preprocess(img, dim = (256,256)):
        img = img.resize(dim)
        img = np.asarray(img)
        img = (img - 127.5) / 127.5
        return img

def reconstruct(img):
    img = img * 127.5 + 127.5
    return img.astype('uint8')
    
   

 
