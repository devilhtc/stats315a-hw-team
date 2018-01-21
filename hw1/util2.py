from sklearn.neighbors import NearestNeighbors
import numpy as np
import random

class Util():
	def test(self):
		return 'testing complete'

	'''
	KNN wrapper

	inputs:
		X : numpy array (n, 2)
		labels: numpy array (n, ), elements are either 0 or 1 indicating class
		k: int
		query: numpy array (m, 2)

	output:
		queryLabels: numpy array (m, ), elements are either 0 or 1 indicating class
	'''
	def KNNWrapper(self, X, labels, k, query):
		if len(query) == 0:
			return []

		# call sklearn api to get indices
		nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
		distances, indices = nbrs.kneighbors(query)

		# majority vote to produce label for each query
		queryLabels = np.array([1 if 2*np.sum(labels[ele])>k else 0 for ele in indices])
		return queryLabels

	'''
	test KNN accuracy

	inputs:
		X : numpy array (n, 2)
		labels: numpy array (n, ), elements are either 0 or 1 indicating class
		k: int
		query: numpy array (m, 2)
		queryLabels: numpy array (m, ), elements are either 0 or 1 indicating true class

	output:
		accu: float, indicating accuracy 
		
	'''
	def testKNNAccuracy(self, X, labels, k, query, queryLabels):
		testQueryLabels = self.KNNWrapper(X, labels, k, query)
		accu = float(np.sum(queryLabels*testQueryLabels)) / float(len(query))

	'''
	generate index of f-fold cross validation

	inputs:
		N: int

	outputs:
		folds: a list (length f) of list of integers (length N/f) indicating the indices for this fold
	'''
	def generateFFoldIndices(self, N, f):
		indices = range(N)
		for i in range(1, N):
			j = random.randint(0, i - 1)
			indices[i], indices[j] = indices[j], indices[i]
		folds = [[] for _ in range(f)]
		for i in range(N):
			folds[i%f].append(indices[i])
		return folds

	def fFoldCrossValidation(self, X, labels, f):
		foldIndices = self.generateFFoldIndices(len(labels), f)



