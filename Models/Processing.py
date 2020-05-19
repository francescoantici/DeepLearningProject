import numpy as np
from PIL import Image
from math import ceil


def preprocess(img, dim = (256, 256)):
        img = img.resize(dim)
        img = np.asarray(img)
        img = (img - 127.5) / 127.5
        return img

def patcher(image, shape = (256, 256), stride = (256 - 24, 256)):
        out = []
        image = np.asarray(image)
        image = (image - 127.5) / 127.5
        for row in range(0, image.shape[0]-24, stride[0]):
                for column in range(0, image.shape[1], stride[1]):
                        out.append(image[row:row+shape[0], column:column+shape[1], :])
        return np.asarray(out).reshape((len(out), shape[0], shape[1], 3))

def reconstruct(patches, shape = (720, 1280), deprocess = True):
        out = []
        batch = int(patches.shape[0] / int((shape[0] * shape[1]) / (patches.shape[1] * patches.shape[2])))
        columns = int(shape[1] / patches.shape[2])
        rows = ceil( shape[0] / patches.shape[1])
        img  = Image.fromarray(np.zeros((shape[0], shape[1], 3)), 'RGB')
        for row in range(rows):
                for column in range(columns):
                        if deprocess:
                                img.paste(Image.fromarray(deprocess_image(patches[columns*row + column]), 'RGB') , (column*patches.shape[2], row*(patches.shape[1] - 24) ))
                        else:
                                img.paste(Image.fromarray(patches[columns*row + column], 'RGB') , (column*patches.shape[2], row*(patches.shape[1] - 24) ))
        out.append(np.asarray(img))
        
        return np.asarray(out).astype('uint8').reshape((shape[0], shape[1], 3))
    
def deprocess_image(img):
        img = img * 127.5 + 127.5
        return img.astype('uint8')


