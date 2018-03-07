import numpy as np
from q6 import get_data
from sklearn import svm
from sklearn.model_selection import cross_val_score

def svm_cv(X, y):
    y = np.reshape(y, [-1])
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
        svm_clf = svm.SVC(kernel=kernel)
        scores = cross_val_score(svm_clf, X, y, cv=5)
        print "SVM model kernel: " + kernel + ", cross validation score:"
        print np.mean(scores)

def main():
    X_train, y_train, _ = get_data()

    # SVM cross validation
    svm_cv(X_train, y_train)

if __name__ == '__main__':
    main()