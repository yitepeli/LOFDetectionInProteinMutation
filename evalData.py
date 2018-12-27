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
import random
import sklearn.naive_bayes as nb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import blossomProcessor as bloPro


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

#-----Report and Accuracy Parts------#


def Clf_Report(y_test, y_pred, algorithmName):
    accuracyScore = accuracy_score(y_test, y_pred)
    print("---------------------TEST RESULTS---------------------\n")
    print("       Classifier : ", algorithmName)
    print("       Accuracy   : ", accuracyScore*100, "\n")
    print(classification_report(y_test, y_pred))

def Clf_BaseAccuracy(labelsTest):
    # Calculate Base Accuracy
    countOfLoss = 0
    sizeOfTest = len(labelsTest)
    for row in labelsTest:
        if row == 1:
            countOfLoss += 1
    baseAcc = max(countOfLoss, sizeOfTest - countOfLoss) / sizeOfTest
    print("Base Accuracy: " + str(baseAcc))

def Clf_TPFP(testPredictions, labelsTest):
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
    F1 = 2 * precision * recall / (precision + recall)
    print("Accuracy: " + str(acc * 100))
    print("Precision: " + str(precision * 100))
    print("Recall: " + str(recall * 100))
    print("F1 Measure: " + str(F1 * 100))

    '''countOfCorrects = 0
    for i in range(len(labelsTest)):
        if testPredictions[i] == labelsTest[i]:
            countOfCorrects += 1
    predictionAcc = countOfCorrects / sizeOfTest
    print("Prediction Accuracy: " + str(predictionAcc))
    '''

def Clf_CompareLabels(y_preds, y_test):
    print("\n---------COMPARING LABELS---------\n")
    for i in range(0, len(y_test)):
        print("predict --> ", y_preds[i], "actual -->", y_test[i])

def Clf_Split_Data():
    data, labels = processData()

    # Encode amino acids
    oneHotData = convertToOneHot(data)
    # print(OneHotData)

    x, y = addFeatures(data, "Data/Amino Acids Properties.csv", 4)
    oneHotData = np.append(oneHotData, x, axis=1)

    #data split operation based on stratified labels. %80 train %20 test (rate could be changeable)
    oneHotDataTrain, oneHotDataTest, labelsTrain, labelsTest = train_test_split(oneHotData, labels, stratify=labels, test_size=0.20, random_state=50)

    # Train and Test Data 80%-20%
    # cutterIndex = round(8 * len(labels) / 10)
    # oneHotDataTrain = oneHotData[:cutterIndex]
    # oneHotDataTest = oneHotData[cutterIndex:]
    # labelsTrain = labels[:cutterIndex]
    # labelsTest = labels[cutterIndex:]
    # sizeOfTrain = len(labelsTrain)
    # sizeOfTest = len(labelsTest)

    return oneHotDataTrain, oneHotDataTest, labelsTrain, labelsTest

def Clf_RandomForest(x_train, x_test, y_train):
    clf = RandomForestClassifier(max_depth=100, n_estimators=100)
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)
    return y_preds

def Clf_DecisionTree(x_train, x_test, y_train):
    clf = tree.DecisionTreeClassifier(max_depth=1000)
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)
    return y_preds

def Clf_LogisticRegression(x_train, x_test, y_train):
    clf = LogisticRegression(C=0.01, max_iter=1000)
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)
    return y_preds

def Clf_SVM(x_train, x_test, y_train, kernelType):
    if(kernelType == "rbf"):
        clf = svm.SVC(kernel="rbf", gamma="auto")
    if(kernelType == "linear"):
        clf = svm.SVC(kernel="linear", C=0.01, max_iter=1000)

    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)
    return y_preds

def Clf_SGDC(x_train, x_test, y_train):
    clf = SGDClassifier(alpha=0.001)
    clf.fit(x_train, y_train)
    y_preds = clf.predict(x_test)
    return y_preds

def Clf_KNN(x_train, x_test, y_train):
    kNN = KNeighborsClassifier(n_neighbors=5)
    kNN.fit(x_train, y_train)
    y_pred = kNN.predict(x_test)
    return y_pred

def Clf_GaussianNB(x_train, x_test, y_train):
    clf = nb.GaussianNB()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


#-----Main Parts------#
'''def main():
    #split and get train and test features and labels
    x_train, x_test, y_train, y_test = Clf_Split_Data()
    Clf_BaseAccuracy(y_test)

    # y_RF_preds = Clf_RandomForest(x_train, x_test, y_train)
    # Clf_Report(y_test, y_RF_preds, "Random Forest")

    y_SVM_preds = Clf_SVM(x_train, x_test, y_train, "linear")
    Clf_Report(y_test, y_SVM_preds, "Support Vector Machine - Linear Kernel with C=0.01, Gamma=auto")

    #y_SVM_preds = Clf_SVM(x_train, x_test, y_train, "rbf")
    #Clf_Report(y_test, y_SVM_preds, "Support Vector Machine - RBF Kernel")
    # Clf_TPFP(y_SVM_preds, y_test)

    # y_SGDC_preds = Clf_SGDC(x_train, x_test, y_train)
    # Clf_Report(y_test, y_SGDC_preds, "Stochastic Gradient Descent Classifier")
    # Clf_TPFP(y_SGDC_preds, y_test)

    # y_KNN_preds = Clf_SGDC(x_train, x_test, y_train)
    # Clf_Report(y_test, y_KNN_preds, "K-Nearest Neighbor")
    # Clf_TPFP(y_KNN_preds, y_test)

    # y_GNB_preds = Clf_SGDC(x_train, x_test, y_train)
    # Clf_Report(y_test, y_GNB_preds, "Gaussian Naive Bayes")
    # Clf_TPFP(y_GNB_preds, y_test)


    #shows actual label and predicted label
    #Clf_CompareLabels(y_SVM_preds, y_test)

'''
def predict():
    data, labels = processData()

    # Encode amino acids
    oneHotData = convertToOneHot(data)
    # print(OneHotData)

    x,y = addFeatures(data, "Data/Amino Acids Properties.csv",4)
    blosData = getBlossomData(data)

    oneHotData = np.append(oneHotData, blosData, axis=1)

    # Train and Test Data 80%-20%
    '''cutterIndex = round(8 * len(labels) / 10)
    oneHotDataTrain = oneHotData[:cutterIndex]
    oneHotDataTest = oneHotData[cutterIndex:]
    labelsTrain = labels[:cutterIndex]
    labelsTest = labels[cutterIndex:]
    sizeOfTrain = len(labelsTrain)
    sizeOfTest = len(labelsTest)'''
    oneHotDataTrain, oneHotDataTest, labelsTrain, labelsTest = train_test_split(oneHotData, labels, stratify=labels,
                                                                                test_size=0.20, random_state=50)
    sizeOfTest = len(labelsTest)
    # Calculate Base Accuracy
    countOfLoss = 0
    for row in labelsTest:
        if row == 1:
            countOfLoss += 1
    baseAcc = max(countOfLoss, sizeOfTest - countOfLoss) / sizeOfTest
    print("Base Accuracy: " + str(baseAcc))

    # Create SVM Model
    clf = SVC(kernel= "linear", C=0.01, gamma='auto')
    clf.fit(oneHotDataTrain, labelsTrain)

    # Predict
    testPredictions = clf.predict(oneHotDataTest)

    # Prediction Statistics
    '''TP = 0
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
    F1 = 2*precision*recall/(precision+recall)
    print("Accuracy:" + str(acc))
    print("Precision:" + str(precision))
    print("Recall:" + str(recall))
    print("F1 Measure:" + str(F1))'''
    accuracyScore = accuracy_score(labelsTest, testPredictions)
    print("---------------------TEST RESULTS---------------------\n")
    print("       Classifier : SVM-Linear C=0.01")
    print("       Accuracy   : ", accuracyScore * 100, "\n")
    print(classification_report(labelsTest, testPredictions))

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