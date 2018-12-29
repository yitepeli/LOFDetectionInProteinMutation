# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:42:39 2018

@author: yitepeli
"""

import csv
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import blossomProcessor as bloPro
import classification
import evaluation


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

    c = np.c_[outData.reshape(len(outData), -1), outLabel.reshape(len(outLabel), -1)]
    random.shuffle(c)
    a2 = c[:, :outData.size // len(outData)].reshape(outData.shape)
    a2 = np.array(a2)
    b2 = c[:, outData.size // len(outData):].reshape(outLabel.shape)
    b2 = np.array(b2)
    return outData, outLabel


def convertToOneHot(dataInput):
    df = pd.DataFrame({'A': dataInput[:, 0].tolist(), 'B': dataInput[:, 1].tolist()})
    dataOutput = np.array(pd.get_dummies(df, prefix=['aa0', 'aa1']).values.tolist())
    return dataOutput

def addFeatures(labels, inFile, noOfCols):
    normMax = [-999]*(noOfCols-1)
    normMin = [999]*(noOfCols-1)

    matrixForm = []
    with open(inFile, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        #For each row in file
        for row in spamreader:
            temp = []
            for colNo in range(0,noOfCols):
                temp.append(row[colNo])
            matrixForm.append(temp)



    matrixForm = matrixForm[1:][:]
    for row in matrixForm:
        for colNo in range(1,len(row)):
            if float(row[colNo]) > normMax[colNo-1]:
                normMax[colNo - 1] = float(row[colNo])
            if float(row[colNo]) < normMin[colNo-1]:
                normMin[colNo - 1] = float(row[colNo])

    for row in matrixForm:
        for colNo in range(1,len(row)):
            row[colNo] = (float(row[colNo])-normMin[colNo-1])/(normMax[colNo-1]-normMin[colNo-1])

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

def getBlossomData(aaLabels):
    blossomDataList, blossomDataScore = bloPro.processData()


    ScoreOutData = np.array([])
    for row in aaLabels:
        ind1 = blossomDataList.index(row[0])
        ind2 = blossomDataList.index(row[1])


        if len(ScoreOutData) == 0:
            ScoreOutData = np.array([[blossomDataScore[ind1][ind2]]])
        else:
            ScoreOutData = np.append(ScoreOutData, [[blossomDataScore[ind1][ind2]]], axis=0)

    return ScoreOutData


def Clf_Split_Data():
    #Get processed data
    data, labels = processData()

    # Encode amino acids
    oneHotData = convertToOneHot(data)
    # print(OneHotData)

    #Get extra features
    x,y = addFeatures(data, "Data/Amino Acids Properties.csv",4)
    blosData = getBlossomData(data)

    #Add extra features
    oneHotData = np.append(oneHotData, blosData, axis=1)

    #data split operation based on stratified labels. %80 train %20 test (rate could be changeable)
    oneHotDataTrain, oneHotDataTest, labelsTrain, labelsTest = train_test_split(oneHotData, labels, stratify=labels, test_size=0.20, random_state=50)

    return oneHotDataTrain, oneHotDataTest, labelsTrain, labelsTest


def predict():

    #split Train, Test Data
    oneHotDataTrain, oneHotDataTest, labelsTrain, labelsTest = Clf_Split_Data()

    #Use Model
    testPredictions = classification.Clf_SVM(oneHotDataTrain, oneHotDataTest, labelsTrain, "rbf")

    #Report
    evaluation.Clf_Report(labelsTest, testPredictions, "SVM with RBF Kernel")

def main():
    predict()

if __name__== "__main__":
    main()