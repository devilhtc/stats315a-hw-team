import numpy as np
from q6 import get_data
import sklearn.linear_model as sklm
from sklearn import svm
from utils import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def preprocess(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

def main():
    X_train, y_train, X_test = get_data()
    X_train = preprocess(X_train)
    model = sklm.LogisticRegression()
    model.fit(X_train, y_train)
    scores = model.predict(X_test)
    print(confidence_interval(scores))

if __name__ == '__main__':
    main()
