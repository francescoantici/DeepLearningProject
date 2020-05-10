import numpy as np
from PIL import Image
from scipy.signal import deconvolve
from cv2 import filter2D
from random import sample

def patcher(image, shape, stride = None, n_patches = None):
    out = []
    if not stride:
        stride = shape
    for row in range(0, image.shape[0], stride[0]):
        for column in range(0, image.shape[1], stride[1]):
            out.append(image[row:row+shape[0], column:column+shape[1], :])
    return np.asarray(out).reshape((len(out), shape[0], shape[1], 3))

def reconstruct(patches, shape, kernel = None):
    out = []
    batch = int(patches.shape[0] / int((shape[0] * shape[1]) / (patches.shape[1] * patches.shape[2])))
    columns = int(shape[1] / patches.shape[2])
    rows = int( shape[0] / patches.shape[1])
    for i in range(0, patches.shape[0], batch): 
        img  = Image.fromarray(np.zeros((shape[0], shape[1], 3)), 'RGB')
        for row in range(rows):
            for column in range(columns):
                img.paste(Image.fromarray(patches[columns*row + column], 'RGB'),(column*patches.shape[2], row*patches.shape[1]))
        out.append(np.asarray(img))
    return np.asarray(out)
    #return deconvolve(np.asarray(out), kernel)

 
