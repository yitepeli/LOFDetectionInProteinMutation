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
    with open('Data/PPData.csv', newline='') as csvfile:
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


def convertToOneHot(dataInput):
    df = pd.DataFrame({'A': dataInput[:, 0].tolist(), 'B': dataInput[:, 1].tolist()})
    dataOutput = np.array(pd.get_dummies(df, prefix=['aa0', 'aa1']).values.tolist())
    return dataOutput

def addFeatures(labels, inFile):
    matrixForm = []
    with open(inFile, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        #For each row in file
        for row in spamreader:
            temp = []
            for colNo in range(0,4):
                temp.append(row[colNo])
            matrixForm.append(temp)

    print(matrixForm)
    n = np.array(matrixForm)
    indexF = n[:,0]

    outData = np.array([])
    outLabel = np.array([])

    # For each row in file for left column
    for row in labels:
        i1 = indexF.tolist().index(row[0])
        i2 = indexF.tolist().index(row[1])
        if len(outLabel) == 0:
            outData = np.array([matrixForm[i1][1:]+matrixForm[i2][1:]])
            outLabel = np.array([row])
        else:
            outData = np.append(outData, [matrixForm[i1][1:]+matrixForm[i2][1:]], axis=0)
            outLabel = np.append(outLabel, [row],axis=0)

    return outData, outLabel

def predict():
    data, labels = processData()

    # Encode amino acids
    oneHotData = convertToOneHot(data)
    # print(OneHotData)

    x,y = addFeatures(data, "Data/Amino Acids Properties.csv")

    oneHotData = np.append(oneHotData,x,axis=1)

    # Train and Test Data 80%-20%
    cutterIndex = round(8 * len(labels) / 10)
    oneHotDataTrain = oneHotData[:cutterIndex]
    oneHotDataTest = oneHotData[cutterIndex:]
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
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for rowNo in range(len(labelsTest)):
        if testPredictions[rowNo] == 1:
            if labelsTest[rowNo] == 1:
                TP = TP + 1
            elif labelsTest[rowNo] == 0:
                FP = FP + 1
        elif testPredictions[rowNo] == 0:
            if labelsTest[rowNo] == 1:
                FN = FN + 1
            elif labelsTest[rowNo] == 0:
                TN = TN + 1
    # print(predictions)

    acc = (TP + TN) / (TP + TN + FP + FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    print("Accuracy:" + str(acc))
    print("Precision:" + str(precision))
    print("Recall:" + str(recall))

    '''countOfCorrects = 0
    for i in range(len(labelsTest)):
        if testPredictions[i] == labelsTest[i]:
            countOfCorrects += 1

    predictionAcc = countOfCorrects / sizeOfTest
    print("Prediction Accuracy: " + str(predictionAcc))
    '''

def main():
    predict()

    
if __name__== "__main__":
    main()