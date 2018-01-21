from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import random

# constants
LOW_LIM = -2
HIGH_LIM = 3
INTERVAL = 0.05

class Util():
	def test(self):
		return 'testing complete'

	'''
		Generate 10 centroids for each class
	'''
	def genCent(self):
		#mean and conv for class 0
		mean_0 = (1, 0); cov_0 = [[1, 0], [0, 1]]
		c_0 = np.random.multivariate_normal(mean_0, cov_0, (10, ))

		mean_1 = (0, 1); cov_1 = [[1, 0], [0, 1]]
		c_1 = np.random.multivariate_normal(mean_1, cov_1, (10, ))
		return (c_0, c_1)

	'''
		cen are centroids
		num_data is the number of datasets
		return result (num_data, 2)
	'''
	def genData(self, cen, num_data):

		result = []
		for _ in range(num_data):
			rand_index = random.randint(0, 9)
			mean = cen[rand_index, :]; cov = [[1.0/5, 0], [0, 1.0/5]]
			data = np.random.multivariate_normal(mean, cov)
			result.append(data)
		return np.stack(result)

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
		nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
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
		accu = float(np.sum(queryLabels == testQueryLabels)) / float(len(query))
		return accu

	'''
	LR wrapper

	inputs:
		X : numpy array (n, 2)
		labels: numpy array (n, ), elements are either 0 or 1 indicating class
		query: numpy array (m, 2)

	output:
		queryLabels: numpy array (m, ), elements are either 0 or 1 indicating class
	'''
	def LRWrapper(self, X, labels, query):
		if len(query) == 0:
			return []

		# call sklearn api to get indices
		regr = LinearRegression()
		regr.fit(X, labels)
		pred = regr.predict(query)

		# majority vote to produce label for each query
		queryLabels = pred > 0.5
		return queryLabels


	'''
	test LR accuracy

	inputs:
		X : numpy array (n, 2)
		labels: numpy array (n, ), elements are either 0 or 1 indicating class
		query: numpy array (m, 2)
		queryLabels: numpy array (m, ), elements are either 0 or 1 indicating true class

	output:
		accu: float, indicating accuracy

	'''
	def testLRAccuracy(self, X, labels, query, queryLabels):
		testQueryLabels = self.LRWrapper(X, labels, query)
		accu = float(np.sum(queryLabels == testQueryLabels)) / float(len(query))
		return accu

	'''
	generate index of f-fold cross validation

	inputs:
		N: int
		f: int

	outputs:
		folds: a list (length f) of np array (length N/f) indicating the indices for this fold
	'''
	def generateFFoldIndices(self, N, f):
		indices = range(N)
		for i in range(1, N):
			j = random.randint(0, i - 1)
			indices[i], indices[j] = indices[j], indices[i]
		folds = [[] for _ in range(f)]
		for i in range(N):
			folds[i%f].append(indices[i])
		for i in range(f):
			folds[i] = np.array(folds[i])
		return folds


	'''
	calculate accuracy of f-fold cross validation with a certain k

	inputs:
		X : numpy array (n, 2)
		labels: numpy array (n, ), elements are either 0 or 1 indicating class
		k: int
		foldIndices: return value from generateFFoldIndices

	outputs:
		accu: float
	'''
	def fFoldCrossValidation(self, X, labels, k, foldIndices):
		f = len(foldIndices)
		accus = []
		for i in range(f):
			trainFolds = [X[foldIndices[j]] for j in range(f) if j != i]
			labelFolds = [labels[foldIndices[j]] for j in range(f) if j != i]
			train = np.concatenate(trainFolds)
			trainLabels = np.concatenate(labelFolds)
			test = X[foldIndices[i]]
			testLabels = labels[foldIndices[i]]
			accus.append(self.testKNNAccuracy(train, trainLabels, k, test, testLabels))
		accu = sum(accus) / float(f)
		return accu

	'''
	generate a 2D mesh

	inputs:
		xlow, xhigh, dx: start, stop and step of x-axis
		ylow, yhigh, dy: start, stop and step of y-axis

	outputs:
		X, Y: meshgrid
	'''
	def generateMesh(self, xlow = LOW_LIM, xhigh = HIGH_LIM, dx = INTERVAL, ylow = LOW_LIM, yhigh = HIGH_LIM, dy = INTERVAL):
		xs = np.arange(xlow, xhigh, dx)
		ys = np.arange(ylow, yhigh, dy)
		X, Y = np.meshgrid(xs, ys)
		return X, Y

	'''
	convert a mesh grid to an array
	'''
	def meshToArray(self, X):
		m, n = X.shape
		xpos = np.reshape(X, (m*n, ))
		return xpos

	'''
	convert an array of size (m*n, ) to a mesh grid
	'''
	def arrayToMesh(self, arr, m, n):
		mesh = np.reshape(arr, (m, n))
		return mesh

	'''
	convert a pair of mesh grid X, Y (both size (m, n)) to a np array of size (m*n, 2)
	'''
	def meshXYToArray(self, X, Y):
		xArray = self.meshToArray(X)
		yArray = self.meshToArray(X)
		xyArray = np.stack([xArray, yArray])
		return xyArray.T

	'''
	convert array of size (m*n, 2) to a XY mesh grid
	'''
	def arrayXYToMesh(self, arr, m, n):
		xArray = arr[:,0]
		yArray = arr[:,1]
		return self.arrayToMesh(xArray, m, n), self.arrayToMesh(yArray, m, n)


	'''
	inputs:
		ks: list of k
		X: training data, (n, 2)
		labels: training labels (n, ) 1 or 0
		quesr: test data, (n, 2)

	outputs:
		X, Y: meshgrid
	'''
	def partB(self, ks, X, labels, query, queryLabels):
		DoF  = [0]*len(ks)
		kNNAccuTrain = [0]*len(ks)
		kNNAccuTest = [0]*len(ks)
		N = float(X.shape[0])
		for i in range(len(ks)):
			k = ks[i]
			print (k)
			DoF[i] = N/k
			kNNAccuTest[i] = self.testKNNAccuracy(X, labels, k, query, queryLabels)
			kNNAccuTrain[i] = self.testKNNAccuracy(X, labels,k,  X, labels)



		#plt.plot(DoF, kNNAccuTest, 'bo', DoF, kNNAccuTrain, 'ro')
		#'''
		plt.plot(DoF, kNNAccuTest, marker='s', linestyle='--', color='C1', label='test')
		plt.plot(DoF, kNNAccuTrain, marker='s', linestyle='--', color='b', label='train')
		plt.ylim(0, 1.0)
		plt.legend()
		#'''
		plt.savefig("./partB.png")
		
	'''
	part C of problem 1

	inputs:
		X : numpy array (n, 2)
		labels: numpy array (n, ), elements are either 0 or 1 indicating class
		f: int, number of folds
		ks: [int]

	output:
		void (but will save the figure for this part)
	'''
	def partC(self, X, labels, f, ks):
		DoF  = []
		N = float(X.shape[0])
		foldIndices = self.generateFFoldIndices(X.shape[0], f)
		accuMeans = []
		accuStds = []
		for k in ks:
			DoF.append(N/k)
			accuMean, accuStd = self.fFoldCrossValidation(X, labels, k, foldIndices)
			accuMeans.append(accuMean)
			accuStds.append(accuStd)
			print k
		plt.figure(1)
		plt.plot(DoF, accuMeans)
		plt.plot(DoF, accuStds)
		#plt.xticks(DoF)
		plt.xlabel('DoF')
		plt.legend(['accu_mean', 'accu_std'])
		plt.savefig('partC.png')