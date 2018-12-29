from sklearn.metrics import accuracy_score, classification_report


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