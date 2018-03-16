import numpy as np
from q6 import get_data
import sklearn.linear_model as sklm
from sklearn import svm
from scipy import stats
from utils import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

import scipy as sp
import scipy.stats

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

def preprocess(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

def mean_confidence_interval(data, confidence=0.90):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

def gaussian_confidence_interval(data, confidence = 0.9):
    N = data.shape[0]
    mu, sigma = np.mean(data), np.std(data)
    return stats.norm.interval(confidence, loc=mu, scale=sigma/np.sqrt(N))

def main():
    X_train, y_train, X_test = get_data()
    #X_train = preprocess(X_train)
    model = sklm.LogisticRegression()
    #model = svm.SVC(C=1, kernel='linear')
    #model = sklm.SGDClassifier(loss = 'squared_hinge', penalty = 'l2')
    #model = ElasticNet(random_state=5)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    print(scores.shape)
    output_to_file(scores)
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    print "The model accuracy is\n", cv(X_train, y_train, model, kf)
    print "\n"
    print "The 90% confidence interval is\n", gaussian_confidence_interval(scores)

if __name__ == '__main__':
    main()
