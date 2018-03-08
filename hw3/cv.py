import numpy as np
from q6 import get_data
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def svm_cv(X, y, kernel, k):
    y = np.reshape(y, [-1])
    scores = []
    Cs = np.logspace(-2, 7, 19)
    for C in Cs:
        print "C = " + str(C)
        svm_clf = svm.SVC(C=C, kernel=kernel)
        kf = KFold(n_splits=k, shuffle=True, random_state=1234)
        scores.append(cv(X, y, svm_clf, kf))
    print Cs
    print scores

def cv(X, y, model, kf):
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = (pred > 0.5)
        accuracy = accuracy_score(y_test, pred)
        #print accuracy
        scores.append(accuracy)
    return np.mean(scores)


def main():
    X_train, y_train, _ = get_data()

    # SVM cross validation
    svm_cv(X_train, y_train, kernel='linear', k=5)

if __name__ == '__main__':
    main()