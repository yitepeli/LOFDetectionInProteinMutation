# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:42:39 2018

@author: yitepeli
@author: omeerkorkmazz
"""

import csv
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import blossomProcessor as bloPro
import classification
import evaluation
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def processData():
    outData = np.array([])
    outLabel = np.array([])
    outSequence = np.array([])

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
                outSequence = np.array([row[0]])
                outData = np.array([[row[1], data2Val]])
                outLabel = np.array([labelVal])
            else:
                outSequence = np.append(outSequence, [row[0]])
                outData = np.append(outData, [[row[1], data2Val]], axis=0)
                outLabel = np.append(outLabel, [labelVal])

    c = np.c_[outData.reshape(len(outData), -1), outLabel.reshape(len(outLabel), -1)]
    random.shuffle(c)
    a2 = c[:, :outData.size // len(outData)].reshape(outData.shape)
    a2 = np.array(a2)
    b2 = c[:, outData.size // len(outData):].reshape(outLabel.shape)
    b2 = np.array(b2)
    return outSequence, outData, outLabel


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

    #print(matrixForm)
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


def getAATableData(aaLabels):
    aaDataList, aaDataScore = bloPro.processDataAATable()


    out1 = np.array([])
    out2 = np.array([])
    for row in aaLabels:
        ind1 = aaDataList.index(row[0])
        ind2 = aaDataList.index(row[1])

        if len(out1) == 0:
            out1 = np.array([aaDataScore[ind1]])
            out2 = np.array([aaDataScore[ind2]])
        else:
            out1 = np.append(out1, [aaDataScore[ind1]], axis=0)
            out2 = np.append(out2, [aaDataScore[ind2]], axis=0)

    return out1, out2

def getEmbeddings(seq,structName):
    sequence, features = bloPro.processDataEmbeddings(structName)

    out1 = np.array([])
    for row in seq:
        if row not in sequence:
            tempArr = [0]*len(features[0])
            out1 = np.append(out1, [tempArr], axis=0)
        else:
            ind1 = sequence.index(row)

            if len(out1) == 0:
                out1 = np.array([features[ind1]])
            else:
                out1 = np.append(out1, [features[ind1]], axis=0)

    return out1


def Clf_Split_Data():
    #Get processed data
    sequence, data, labels = processData()

    # Encode amino acids
    #oneHotData = convertToOneHot(data)
    oneHotData = data
    # print(OneHotData)

    #Get extra features
    x, y = addFeatures(data, "Data/Amino Acids Properties.csv", 4)
    blosData = getBlossomData(data)
    aaData1,aaData2 = getAATableData(data)
    node2vecFeatures1 = getEmbeddings(sequence,"1jm7")
    node2vecFeatures2 = getEmbeddings(sequence,"1t29")

    #Add extra features
    oneHotData = np.append(oneHotData, blosData, axis=1)
    oneHotData = np.append(oneHotData, aaData1, axis=1)
    oneHotData = np.append(oneHotData, aaData2, axis=1)
    oneHotData = np.append(oneHotData, node2vecFeatures1, axis=1)
    oneHotData = np.append(oneHotData, node2vecFeatures2, axis=1)

    #Removing actual aa's
    oneHotData = np.delete(oneHotData, 0, 1)
    oneHotData = np.delete(oneHotData, 0, 1)

    oneHotData = oneHotData.astype(float)

    #data split operation based on stratified labels. %90 train %10 test (rate could be changeable)
    oneHotDataTrain, oneHotDataTest, labelsTrain, labelsTest = train_test_split(oneHotData, labels, stratify=labels, test_size=0.1)

    return oneHotDataTrain, oneHotDataTest, labelsTrain, labelsTest


def predict():

    #split Train, Test Data
    oneHotDataTrain, oneHotDataTest, labelsTrain, labelsTest = Clf_Split_Data()

    #---Use Model---#
    #testPredictions = classification.Clf_SVM(oneHotDataTrain, oneHotDataTest, labelsTrain, "linear")
    #testPredictions = classification.Clf_XGBoost(oneHotDataTrain, oneHotDataTest, labelsTrain)
    #testPredictions = classification.Clf_KNN(oneHotDataTrain, oneHotDataTest, labelsTrain)
    #testPredictions = classification.Clf_DecisionTree(oneHotDataTrain, oneHotDataTest, labelsTrain)
    #testPredictions = classification.Clf_SGDC(oneHotDataTrain, oneHotDataTest, labelsTrain)
    testPredictions = classification.Clf_LogisticRegression(oneHotDataTrain, oneHotDataTest, labelsTrain)
    #testPredictions = classification.Clf_RandomForest(oneHotDataTrain, oneHotDataTest, labelsTrain)

    #---Report---#
    #evaluation.Clf_Report(labelsTest, testPredictions, "XGBoost")
    #evaluation.Clf_Report(labelsTest, testPredictions, "DecisionTree")
    #evaluation.Clf_Report(labelsTest, testPredictions, "SVM with RBF Kernel")
    #evaluation.Clf_Report(labelsTest, testPredictions, "SGDC")
    #evaluation.Clf_Report(labelsTest, testPredictions, "KNN")
    evaluation.Clf_Report(labelsTest, testPredictions, "Logistic Regression")
    #evaluation.Clf_Report(labelsTest, testPredictions, "Random Forest")
    #evaluation.Clf_Report(labelsTest, testPredictions, "Decision Tree")

    #evaluation.Clf_TPFP(labelsTest, testPredictions)
    #evaluation.Clf_CompareLabels(labelsTest, testPredictions)


    #---Tune Parameters---#
    #evaluation.Tune_SVM_Parameters(oneHotDataTrain, labelsTrain)
    #evaluation.Tune_XGBoost_Parameters(oneHotDataTrain, labelsTrain)
    #evaluation.Tune_SGDC_Parameters(oneHotDataTrain, labelsTrain)
    #evaluation.Tune_KNN_Parameters(oneHotDataTrain, labelsTrain)
    #evaluation.Tune_RandomForest_Parameters(oneHotDataTrain, labelsTrain)
    #evaluation.Tune_DecisionTree_Parameters(oneHotDataTrain, labelsTrain)


    #---Validation---#
    #evaluation.Learning_Curve(svm.SVC(kernel="rbf", gamma=0.01, C=100), oneHotDataTrain, labelsTrain, "SVM-Rbf")
    #evaluation.Learning_Curve(svm.SVC(kernel="linear", C=1), oneHotDataTrain, labelsTrain, "SVM-Linear")
    #evaluation.Learning_Curve(tree.DecisionTreeClassifier(max_depth=10, max_leaf_nodes=50), oneHotDataTrain, labelsTrain, 'Decision Tree')
    #evaluation.Learning_Curve(RandomForestClassifier(max_depth=100, n_estimators=100), oneHotDataTrain,labelsTrain, 'Random Forest')
    #evaluation.Learning_Curve(GradientBoostingClassifier(learning_rate=0.01, n_estimators=2000), oneHotDataTrain, labelsTrain,'XGBoost')
    #evaluation.Learning_Curve(KNeighborsClassifier(n_neighbors=7), oneHotDataTrain, labelsTrain, 'KNN')

    #evaluation.Validation_Curve(oneHotDataTrain, labelsTrain)

    #evaluation.KNN_Validation(oneHotDataTrain, labelsTrain)
    # evaluation.Precision_Recall_Curve(labelsTest, testPredictions)

    print("\n------- METRICS -------")
    # evaluation.AUC_Score(labelsTest, testPredictions)
    # evaluation.Average_Precision_Score(labelsTest, testPredictions)
    # evaluation.F1_Score(labelsTest, testPredictions)
    evaluation.Compare_Classifiers(oneHotDataTrain, labelsTrain)


def main():
    predict()

if __name__== "__main__":
    main()