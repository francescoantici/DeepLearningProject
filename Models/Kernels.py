import numpy as np



def kernel_generator():
    return np.load("Models/Kernels.npy", allow_pickle=True)

