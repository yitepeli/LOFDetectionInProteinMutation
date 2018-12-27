import csv
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import itertools
import pandas as pd
import random


def processData():
    aaList = []
    aaVariantData = []
    #matrix = np.array()
    #get data from PPData.csv
    with open('Data/blossom.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        #For each row in file
        for row in spamreader:
            if row[0] == '-':
                aaList = list(row[1:])
            else:
                rowList = list(row[1:])
                aaVariantData.append(rowList)
                # print(rowList)



    return aaList, aaVariantData

    '''c = np.c_[outData.reshape(len(outData), -1), outLabel.reshape(len(outLabel), -1)]
    random.shuffle(c)
    a2 = c[:, :outData.size // len(outData)].reshape(outData.shape)
    a2 = np.array(a2)
    b2 = c[:, outData.size // len(outData):].reshape(outLabel.shape)
    b2 = np.array(b2)
    return outData, outLabel'''

