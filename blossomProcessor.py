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

def processDataAATable():
    aaList = []
    aaVariantData = []
    # matrix = np.array()
    # get data from PPData.csv
    with open('Data/aaTable.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        # For each row in file
        for row in spamreader:
            if row[0] == 'AA':
                print()
            else:
                value = [0,0,0,0]
                for colNo in range(len(row)):
                    if row[colNo] == "Positive" or row[colNo] == "Polar" or row[colNo] == "Aromatic" or row[colNo] == "Large":
                        value[colNo-1] = 1
                    elif row[colNo] == "Negative" or row[colNo] == "Non-Polar" or row[colNo] == "Aliphatic" or row[colNo] == "Small":
                        value[colNo-1] = -1
                aaList.append(row[0])
                rowList = value
                aaVariantData.append(rowList)
                # print(rowList)

    return aaList, aaVariantData

def processDataEmbeddings(structName):
    file = "embeddings/"+structName+".emb"
    sequence = []
    embeddings = []
    # matrix = np.array()
    # get data from PPData.csv
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')

        # For each row in file
        for row in spamreader:
            sequence.append(row[0])
            embeddings.append(row[1:])
            # print(rowList)

    return sequence, embeddings

    '''c = np.c_[outData.reshape(len(outData), -1), outLabel.reshape(len(outLabel), -1)]
    random.shuffle(c)
    a2 = c[:, :outData.size // len(outData)].reshape(outData.shape)
    a2 = np.array(a2)
    b2 = c[:, outData.size // len(outData):].reshape(outLabel.shape)
    b2 = np.array(b2)
    return outData, outLabel'''

