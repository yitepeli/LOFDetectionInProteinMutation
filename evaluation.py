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
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.utils.fixes import signature

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



def Tune_SVM_Parameters(x_train, y_train):
    C_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gamma_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(svm.SVC(kernel="rbf"), param_grid=param_grid, cv=5)
    grid.fit(x_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

def Tune_XGBoost_Parameters(x_train, y_train):
    LR_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    estimators_range = [100, 1000, 2000, 3000]
    param_grid = dict(n_estimators=estimators_range, learning_rate=LR_range)
    grid = GridSearchCV(GradientBoostingClassifier(), param_grid=param_grid, cv=5)
    grid.fit(x_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

def Tune_SGDC_Parameters(x_train, y_train):
    alpha_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    param_grid = dict(alpha=alpha_range)
    grid = GridSearchCV(SGDClassifier(), param_grid=param_grid, cv=5)
    grid.fit(x_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

def Tune_KNN_Parameters(x_train, y_train):
    alpha_range = [3,5,7,9,11,13,15,17,19,21]
    param_grid = dict(n_neighbors=alpha_range)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, cv=5)
    grid.fit(x_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

def Tune_RandomForest_Parameters(x_train, y_train):
    alpha_range = [10,50,100,200,1000]
    estimator_range= [10,100,1000,2000]
    param_grid = dict(max_depth=alpha_range, n_estimators=estimator_range)
    grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
    grid.fit(x_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

def Tune_DecisionTree_Parameters(x_train, y_train):
    alpha_range = [10,50,100,200,1000]
    leaf_range= [10,100,1000,2000]
    param_grid = dict(max_depth=alpha_range, max_leaf_nodes=leaf_range)
    grid = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, cv=5)
    grid.fit(x_train, y_train)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))

def Learning_Curve(clf, x_train, y_train, clfName):
    train_sizes, train_scores, valid_scores = learning_curve(clf, x_train, y_train, train_sizes=[500, 1000, 1494], scoring="accuracy", cv=10)
    plt.title("Learning Curve - " + clfName)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training Score")
    plt.plot(train_sizes, np.mean(valid_scores, axis=1), label="Cross Validation Score", linewidth=2.0, linestyle='dashed')
    plt.xlabel("Data Size")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

def Validation_Curve(x_train, y_train):
    train_scores, valid_scores = validation_curve(svm.SVC(kernel='rbf'), x_train, y_train, param_name="C", param_range=np.logspace(-6, 2, 5), cv=5, scoring="accuracy", n_jobs=1)
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



def AUC_Score(y_test,  y_preds):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_preds)
    score = metrics.auc(fpr, tpr)
    print("AUC Score --->", score * 100)

def Average_Precision_Score(y_test,  y_preds):
    score = metrics.average_precision_score(y_test, y_preds, average="macro")
    print("Average Precision Score --->", score * 100)

def Precision_Recall_Curve(y_test, y_score):
    precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        metrics.average_precision_score(y_test, y_score)))
    plt.show()

def F1_Score(y_test, y_score):
    score= metrics.f1_score(y_test, y_score)
    print("F1 Score --->", score * 100)

def Compare_Classifiers(X, y):
    models = []
    #models.append(('LR', LogisticRegression(C=0.01, max_iter=1000)))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=7)))
    models.append(('Decision Tree', tree.DecisionTreeClassifier(max_depth=10, max_leaf_nodes=50)))
    models.append(('Random Forest', RandomForestClassifier(max_depth=100, n_estimators=100)))
    models.append(('SVM-Rbf', svm.SVC(kernel="rbf", gamma=0.01, C=100)))
    models.append(('SVM-Linear', svm.SVC(kernel="linear", C=1)))
    models.append(('XGBoost', GradientBoostingClassifier(learning_rate=0.01, n_estimators=2000)))
    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        cv_results = cross_val_score(model, X, y, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Classifiers Comparison (CV Scores)')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()