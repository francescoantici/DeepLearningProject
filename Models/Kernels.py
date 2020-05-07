from scipy.io import loadmat
import numpy as np

def kernel_generator():
    return loadmat('Models/kernels.mat')["kernels"].reshape((73,))
        
        



