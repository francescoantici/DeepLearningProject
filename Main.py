import numpy as np
from NeuralNetworkCifar10.Cifar10Loader import Cifar10Loader
from NeuralNetworkCifar10.NeuralNetworkCifar10 import NeuralNetworkCifar10
from NeuralNetworkReds.RedsLoader import RedsLoader
from NeuralNetworkReds.NeuralNetworkReds import NeuralNetworkReds
from PIL import Image
import sys
from scipy.io import loadmat

  
def Main(model = "1", mode = "load", master = "local"):

  
  try:
    NN = CNN[model]["NN"]
    dataLoader = CNN[model]["Loader"]
    weights = CNN[model]["Weights"]
    dataset = CNN[model]["Dataset"][master]
  except:
    raise Exception("The model selected is not valid, please insert [1/2]")


  #DATA COLLECTING PHASE

  #Xtrain, ytrain, Xtest, ytest = Cifar10Loader("NeuralNetworkCifar10/Cifar-10-Dataset").get_train_test() #This line allows us to get a 50 000 sample training set and a 10 000 test set, without any validation set
  #"""
  #Xtrain, ytrain, Xval, yval, Xtest, ytest = Cifar10Loader("NeuralNetworkCifar10/Cifar-10-Dataset").get_Train_Test_Validation()
  arguments = dataLoader(dataset).get_Train_Test_Validation()

  #NEURAL NETWORK INITIALIZATION

  #NN = NeuralNetworkCifar10()

  NN = NN()

  #TRAIN PHASE OR WEIGHTS LOADING

  if mode == "load":
    NN.load_weights(weights) 
  elif mode == "train":
    NN.fit(arguments, epochs = 100)
    NN.save(weights)
  else: 
    raise Exception("The specified method is not valid, please insert train or load")
  

  #NEURAL NETWORK EVALUATION ON TEST SET

  #NN.evaluate(arguments)
  

  #DISPLAY OF THE RESULTS

  #NN.show_different_sigma(Xtest, ytest, [1,2,3])
  #NN.display_sample(arguments)
  #"""
  #Xtrain, ytrain, Xval, yval = RedsLoader("NeuralNetworkReds/Reds-Dataset").get_train_validation()
  #print(yval.shape)

CNN = {
    "1" : {
      "NN" : NeuralNetworkCifar10,
      "Loader" : Cifar10Loader,
      "Weights" : "NeuralNetworkCifar10/Weights/Cifar10Weights.h5",
      "Dataset" : {
        "local" : "NeuralNetworkCifar10/Cifar-10-Dataset",
        "cloud" : "NeuralNetworkCifar10/Cifar-10-Dataset"
      },
    },
    "2" : {
      "NN" : NeuralNetworkReds,
      "Loader" : RedsLoader, 
      "Weights" : "NeuralNetworkReds/Weights/RedsWeights.h5",
      "Dataset" : {
        "local" : "/Users/francesco/Desktop/RedsDataset",
        "cloud" : "/home/francescoantici/dataset/RedsDataset"
      }
    }
  }

if len(sys.argv) == 4:
  model = sys.argv[1]
  mode = sys.argv[2]
  master = sys.argv[3]
else:
  model = "1"
  mode = "load"
  master = "local"
  print("The default configuration will be loaded...")

Main(model, mode, master)
