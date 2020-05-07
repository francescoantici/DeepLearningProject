import pickle
import numpy as np


class Loader:

    def __init__(self,link):
        self._url=link

    #Decode the dataset
    def _uncrypt(self, files):
        data = {}
        for doc in files:
            url=self._url+"/"+doc
            try:
                loader = open(url, 'rb')
                data[doc] = pickle.load(loader, encoding='bytes')
            except Exception:
                print("Impossible to read file {}".format(doc))
        return data
    
    def _getTrain(self):
        return {}
    
    def _getTest(self):
        return {}
    
    def get_Train_Test_Validation(self):
        return self._get_Train_Test_Validation()