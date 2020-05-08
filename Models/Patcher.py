import numpy as np
from PIL import Image
from scipy.signal import deconvolve
from cv2 import filter2D

def patcher(image, shape):
    out = []
    for row in range(0, image.shape[0], shape[0]):
        for column in range(0, image.shape[1], shape[1]):
            out.append(image[row:row+shape[0], column:column+shape[1], :])
    return np.asarray(out).reshape((len(out), shape[0], shape[1], 3))

def reconstruct(patches, shape, kernel):
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
    #return np.asarray(out)
    return deconvolve(np.asarray(out), kernel)

kernels = np.load("Models/Kernels.npy", allow_pickle=True)
pic = np.asarray(Image.open("/Users/francesco/Desktop/RedsDataset/train_sharp/230/00000005.png"))
blurred = filter2D(pic, -1, kernels[2])
img1 = Image.fromarray(blurred, 'RGB')
img1.show()
img = np.polydiv(blurred,kernels[2])
img2 = Image.fromarray(img, 'RGB')
img2.show()


        

