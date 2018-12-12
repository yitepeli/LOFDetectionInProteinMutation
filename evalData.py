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

def main():

    data = np.array([])
    label = np.array([])
    with open('PPData.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            #print(row[6])
            labelVal=0
            data2Val=row[2]
            if row[6]=="LOF":
                labelVal=1
            if row[2]=="*":
                data2Val="X"
            if len(label)==0:
                data = np.array([[row[1],data2Val]])
                label = np.array([labelVal])                
            else:
                data = np.append(data, [[row[1],data2Val]],axis = 0) 
                label = np.append(label, [labelVal]) 
    #print(data)
    #print(data[:,0])
    c = itertools.chain(data[:,0], data[:,1])
    anyFeature = list(set(list(c)))
    #d = data[:,0]+data[:,1]
    #anyFeature = list(set(data[:,0]+data[:,1]))
    #print(anyFeature)
    #list(le.classes_)

    enc = preprocessing.LabelEncoder()
    enc.fit(anyFeature)  
    newFeatures = enc.transform(anyFeature)
    print(anyFeature)
    print(list(newFeatures))
    print(data)
    #print(list(enc.classes_))
    encData = np.array([])
    for rowNo in range(len(data)):
        ind1 = anyFeature.index(data[rowNo][0])
        ind2 = anyFeature.index(data[rowNo][1])
        if len(encData)==0:
            encData = np.array([[newFeatures[ind1],newFeatures[ind2]]])      
        else:
            encData = np.append(encData, [[newFeatures[ind1],newFeatures[ind2]]],axis = 0) 
    #enc.transform(data)
    
    print(encData)
    
    clf = SVC(gamma='auto')
    clf.fit(encData, label) 
    #print(list(enc.inverse_transform([1,9])))
    count=0
    '''for i in range(len(label)):
        if clf.predict([encData[i]]) == [1]:
           count += 1
    print(len(label))'''
    print(data[0])
    print(encData[0])
    print(clf.predict([encData[0]]))
    #preprocessing.OneHotEncoder()

    
if __name__== "__main__":
    main()