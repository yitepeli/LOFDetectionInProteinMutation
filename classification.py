import sklearn.naive_bayes as nb
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
from sklearn.ensemble import RandomForestClassifier

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
