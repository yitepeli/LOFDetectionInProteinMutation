# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:42:39 2018

@author: yitepeli
"""

import csv
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import itertools
import pandas as pd


def processData():
    outData = np.array([])
    outLabel = np.array([])

    #get data from PPData.csv
    with open('PPData.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        #For each row in file
        for row in spamreader:
            labelVal = 0
            data2Val = row[2]

            #If LOF then label=1, else label = 0
            if row[6] == "LOF":
                labelVal = 1

            #If new aa is * then it is converted to X
            if row[2] == "*":
                data2Val = "X"
            if len(outLabel) == 0:
                outData = np.array([[row[1], data2Val]])
                outLabel = np.array([labelVal])
            else:
                outData = np.append(outData, [[row[1], data2Val]], axis=0)
                outLabel = np.append(outLabel, [labelVal])

    return outData, outLabel

def predict():
    data, labels = processData()

    # Encode amino acids
    df = pd.DataFrame({'A': data[:, 0].tolist(), 'B': data[:, 1].tolist()})
    OneHotData = np.array(pd.get_dummies(df, prefix=['aa0', 'aa1']).values.tolist())
    # print(OneHotData)

    # Train and Test Data 80%-20%
    cutterIndex = round(8 * len(labels) / 10)
    oneHotDataTrain = OneHotData[:cutterIndex]
    oneHotDataTest = OneHotData[cutterIndex:]
    labelsTrain = labels[:cutterIndex]
    labelsTest = labels[cutterIndex:]
    sizeOfTrain = len(labelsTrain)
    sizeOfTest = len(labelsTest)

    # Calculate Base Accuracy
    countOfLoss = 0
    for row in labelsTest:
        if row == 1:
            countOfLoss += 1
    baseAcc = max(countOfLoss, sizeOfTest - countOfLoss) / sizeOfTest
    print("Base Accuracy: " + str(baseAcc))

    # Create SVM Model
    clf = SVC(gamma='auto')
    clf.fit(oneHotDataTrain, labelsTrain)

    # Predict
    testPredictions = clf.predict(oneHotDataTest)

    # Prediction Statistics
    countOfCorrects = 0
    for i in range(len(labelsTest)):
        if testPredictions[i] == labelsTest[i]:
            countOfCorrects += 1

    predictionAcc = countOfCorrects / sizeOfTest
    print("Prediction Accuracy: " + str(predictionAcc))

def main():
    predict()
    
if __name__== "__main__":
    main()