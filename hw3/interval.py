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

import scipy as sp
import scipy.stats



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
    X_train = preprocess(X_train)
    model = sklm.LogisticRegression()
    #model = svm.SVC(C=1, kernel='linear')

    model.fit(X_train, y_train)
    scores = model.predict(X_test)
    print(gaussian_confidence_interval(scores))

if __name__ == '__main__':
    main()
