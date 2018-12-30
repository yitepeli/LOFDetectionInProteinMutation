from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import sklearn.naive_bayes as nb
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score


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



def CrossVal(x_train, y_train):
    C_range = np.logspace(-2, 2, 3)
    gamma_range = np.logspace(-9, 3, 3)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(svm.SVC(kernel="rbf"), param_grid=param_grid, cv=5)
    grid.fit(x_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))


def Learning_Curve(clf, x_train, y_train, clfName):
    train_sizes, train_scores, valid_scores = learning_curve(clf, x_train, y_train, train_sizes=[250, 500, 750, 1000, 1250, 1500, 1750], cv=5, n_jobs=1, scoring="accuracy")
    plt.title("Learning Curve - " + clfName)
    plt.plot(train_sizes, train_scores, label="Training Score")
    plt.plot(train_sizes, valid_scores, label="Cross Validation Score", linewidth=2.0, linestyle='dashed')
    plt.xlabel("Data Size")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


def Validation_Curve(x_train, y_train):
    train_scores, valid_scores = validation_curve(svm.SVC(), x_train, y_train, param_name="C", param_range=np.logspace(-6, 2, 5), cv=5, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)

    plt.title("Validation Curve with SVM")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(np.logspace(-6, 2, 5), train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(np.logspace(-6, 2, 5), train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(np.logspace(-6, 2, 5), test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(np.logspace(-6, 2, 5), test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()





def KNN_Validation(x_train, y_train):
    x_values = [1, 2, 3, 4, 5]
    neighborValues = []
    averageCV = []
    plt.title("KNN Cross-Validation Scores with different neighbors ")
    for i in range(0, 50):
        if i % 2 == 0:
            clf = KNeighborsClassifier(n_neighbors=i + 1)
            scores = cross_val_score(clf, x_train, y_train, cv=5)
            averageCV.append(sum(scores) / len(scores))
            neighborValues.append(i+1)
        else:
            clf = KNeighborsClassifier(n_neighbors=i)
            scores = cross_val_score(clf, x_train, y_train, cv=5)
            averageCV.append(sum(scores) / len(scores))
            neighborValues.append(i)


    plt.xlabel("Neighbors")
    plt.ylabel("Average Score of Cross Validation")
    plt.plot(neighborValues, averageCV)
    plt.show()