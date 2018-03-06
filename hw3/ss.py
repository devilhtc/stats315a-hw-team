from __future__ import print_function
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC


from q6 import get_data
from models import LinearModel

import unittest

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

print(X_iris.shape)

def var_threshold(X, thres = (.8 * (1 - .8))):
    #select features with variance > thres
    sel = VarianceThreshold(threshold = thres)
    return sel.fit_transform(X)


def univariate_feature_selection(X, y, nums):
    #nums means how many features we wan t to keep
    X_new = SelectKBest(chi2, k = nums).fit_transform(X, y)
    return X_new

def l1_selection(X, y):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    return model.transform(X)


class SSTests(unittest.TestCase):
    def setUp(self):
        self.X_train, self.Y_train = X_iris, y_iris

    def test_1(self):
        X_new = var_threshold(self.X_train)
        self.assertEqual(np.shape(X_new)[0], np.shape(self.X_train)[0], \
        'var_threshold failed')

    def test_2(self):
        X_new = univariate_feature_selection(self.X_train, self.Y_train, 2)
        #print(X_new.shape)
        self.assertEqual(np.shape(X_new)[1], 2, \
        'univariate_feature_selection failed')

    def test_3(self):
        X_new = l1_selection(self.X_train, self.Y_train)
        self.assertEqual(np.shape(X_new)[0], np.shape(self.X_train)[0], \
        'l1_selection failed')

def main():
    unittest.main()

if __name__ == '__main__':
    main()
