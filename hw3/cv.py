import numpy as np
from q6 import get_data
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def svm_cv(X, y):
    y = np.reshape(y, [-1])
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
        svm_clf = svm.SVC(kernel=kernel)
        cv(X, y, svm_clf, 5)

def cv(X, y, model, k):
    kf = KFold(n_splits=k, shuffle=True)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print("start fitting")
        model.fit(X_train, y_train)
        print("finish fitting")
        pred = model.predict(X_test)
        pred = (pred > 0.5) * 1
        accuracy = accuracy_score(y_test, pred)
        print accuracy


def main():
    X_train, y_train, _ = get_data()

    # SVM cross validation
    svm_cv(X_train, y_train)

if __name__ == '__main__':
    main()