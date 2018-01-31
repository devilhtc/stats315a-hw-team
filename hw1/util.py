from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import random

SEED = 985
random.seed(SEED)
np.random.seed( random.randint(0, SEED) + SEED ** 2 )

# constants
LOW_LIM = -2.0
HIGH_LIM = 3.0
INTERVAL = 0.05

class Util():
	def test(self):
		return 'testing complete'

	'''
		Generate 10 centroids for each class
	'''
	def genCent(self):
		# mean and conv for class 0
		mean_0 = (1, 0); cov_0 = [[1, 0], [0, 1]]
		c_0 = np.random.multivariate_normal(mean_0, cov_0, (10, ))

		mean_1 = (0, 1); cov_1 = [[1, 0], [0, 1]]
		c_1 = np.random.multivariate_normal(mean_1, cov_1, (10, ))
		return (c_0, c_1)

	'''
		#### PART A ####
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
	generatea n data points and label for each set of centroids

	inputs:
		centroids: a tuple (centroids_0, centroids_1), centroids_i is a np array (10, 2) or centroid locations
		n: number of data points for each of the two classes

	outputs:
		X: np array (n, 2)
		labels: np array (n, ) of 0/1 labels
	'''
	def genDataAndLabel(self, centroids, n):
		c0, c1 = centroids
		data1 = self.genData(c0, n)
		data2 = self.genData(c1, n)
		X = np.concatenate((data1, data2))
		labels = np.asarray([0]*n + [1]*n)
		return X, labels

	'''
	KNN wrapper

	inputs:
		X : numpy array (n, 2)
		labels: numpy array (n, ), elements are either 0 or 1 indicating class
		k: int
		query: numpy array (m, 2)

	output:
		queryPrediction: numpy array (m, ), elements range from 0 to 1 indicating the probability that it belongs to class 1
	'''
	def KNNWrapper(self, X, labels, k, query):
		if len(query) == 0:
			return []

		# call sklearn api to get indices
		nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
		distances, indices = nbrs.kneighbors(query)

		# return the proportion of label 1
		queryPredictions = np.array([float(np.sum(labels[ele]))/float(k) for ele in indices])
		return queryPredictions

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
		testQueryPredictions = self.KNNWrapper(X, labels, k, query)
		testQueryLabels = np.array(testQueryPredictions > 0.5, dtype = np.int64)
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

		
		queryLabels = pred >= 0.5
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
		return np.mean(accus), np.std(accus)

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
		yArray = self.meshToArray(Y)
		xyArray = np.stack([xArray, yArray])
		return xyArray.T

	'''
	convert array of size (m*n, 2) to a XY mesh grid
	'''
	def arrayXYToMesh(self, arr, m, n):
		xArray = arr.T[0]
		yArray = arr.T[1]
		return self.arrayToMesh(xArray, m, n), self.arrayToMesh(yArray, m, n)

	'''
	convert a list of accuracy to a list of error
	'''
	def accuToErr(self, accus):
		return [1.0 - accu for accu in accus]

	'''
	calculate the probability that point belongs to centroids (with sigma = 0.2)

	inputs:
		point: np array (2, )
		centroids: np array (10, 2)

	output:
		p: sum of probabilities
	'''
	def pFromCentroids(self,point, centroids):
		p = 0.0
		sigma = 0.2
		denomenator = 2.0 * sigma ** 2
		for c in centroids:
			p += np.exp( -np.sum(np.square(point - c)) / denomenator)
		#print p
		return p

	'''
	get the label of all points from points and centroids

	inputs:
		point: np array (m*n, )
		a tuple (centroids_0, centroids_1), centroids_i is a np array (10, 2) or centroid locations

	output:
		preds: np array (m*n, ) with elements between 0 and 1 indicating the probability that it belongs to class 1
	'''
	def predsFromCentroids(self, points, centroids):
		preds = []
		for point in points:
			p0 = self.pFromCentroids(point, centroids[0])
			p1 = self.pFromCentroids(point, centroids[1])
			preds.append( p1 / (p1 + p0) )
		return np.array(preds)

	'''
	inputs:
		ks: list of k
		X: training data, (n, 2)
		labels: training labels (n, ) 1 or 0
		query: test data, (n, 2)
		queryLabels: test labels, (n, ) 1 or 0

	outputs:
		void (but will save the figure for this part)
	'''
	def partB(self, ks, X, labels, query, queryLabels):
		print('Calculating part B ... ')
		DoF  = []
		kNNAccuTrain = []
		kNNAccuTest = []
		N = float(X.shape[0])

		for k in ks:
			print( 'k = '+str(k) )
			DoF.append(N / k)
			kNNAccuTest.append( self.testKNNAccuracy(X, labels, k, query, queryLabels) )
			kNNAccuTrain.append( self.testKNNAccuracy(X, labels,k,  X, labels) )

		LRAccuTest = self.testLRAccuracy(X, labels, query, queryLabels)
		LRAccuTrain = self.testLRAccuracy(X, labels, X, labels)



		# plot two lines: train and test
		plt.plot(DoF, self.accuToErr(kNNAccuTest), marker='s', linestyle='--', color='C1', label='test')
		plt.plot(DoF, self.accuToErr(kNNAccuTrain), marker='s', linestyle='--', color='b', label='train')
		plt.scatter(75, 1-LRAccuTest, label = 'linear test')
		plt.scatter(75, 1-LRAccuTrain, label = 'linear train')


		# set plot parameters
		plt.ylim(0, .6)
		plt.xlabel('DoF (N/k)')
		plt.ylabel('Error rate')
		plt.legend()
		plt.savefig("partB.png")

		print('figure saved')
		print

	'''
	part C of problem 1

	inputs:
		X : numpy array (n, 2)
		labels: numpy array (n, ), elements are either 0 or 1 indicating class
		f: int, number of folds
		ks: list of k

	output:
		kOpt: the k that gives maximum average accuracy
	'''
	def partC(self, X, labels, f, ks):
		print('Calculating part C ... ')
		DoF  = []
		accuMeans = []
		accuStds = []
		N = float(X.shape[0])

		foldIndices = self.generateFFoldIndices(X.shape[0], f)

		for k in ks:
			print( 'k = '+str(k) )
			accuMean, accuStd = self.fFoldCrossValidation(X, labels, k, foldIndices)
			DoF.append(N / k)
			accuMeans.append(accuMean)
			accuStds.append(accuStd)

		plt.figure(1)

		# generate figure and subplot
		fig, ax1 = plt.subplots()

		# first part, plot means
		ax1.plot(DoF, accuMeans, marker='s', linestyle='--', color='C1', label='accu_mean')
		ax1.set_xlabel('DoF (N/k)')
		ax1.set_ylabel('accu_mean', color='C1')
		ax1.tick_params('y', colors='C1')

		# second part, plot stds
		ax2 = ax1.twinx()
		ax2.plot(DoF, accuStds, marker='s', linestyle='--', color='b', label='accu_std')
		ax2.set_ylabel('accu_std', color='b')
		ax2.tick_params('y', colors='b')

		plt.savefig('partC.png')

		print('figure saved')
		print

		# return the k with max accu mean
		maxIndex = max([(accuMeans[i], i) for i in range(len(accuMeans))])[1]
		return ks[maxIndex]

	'''
	part D of problem 1

	inputs:
		train : numpy array (n, 2) (renamed from X to avoid confusion with meshgrid)
		trainLabels: numpy array (n, ), elements are either 0 or 1 indicating class
		k: optimal k found in part C
		centroids: a tuple (centroids_0, centroids_1), centroids_i is a np array (10, 2) or centroid locations

	output:
		void (but will save the figure for this part)
	'''
	def partD(self, train, trainLabels, k, centroids):
		print('Calculating part D ... ')

		numTrain = train.shape[0]
		trainPos0 = train[:numTrain/2].T
		trainPos1 = train[numTrain/2:].T

		# generate array
		X, Y = self.generateMesh()
		xyArray = self.meshXYToArray(X, Y)

		m, n = X.shape
		# get prediction from KNN
		knnPredsArray = self.KNNWrapper(train, trainLabels, k, xyArray)
		knnPreds = self.arrayToMesh(knnPredsArray, m, n)

		# get prediction from Bayes
		bayesPredsArray = self.predsFromCentroids(xyArray, centroids)
		bayesPreds = self.arrayToMesh(bayesPredsArray, m, n)

		# plotting
		plt.figure(3, figsize=(8, 8))

		# use -0.1 and 1.1 as dummy contour levels
		lvls = np.array([ 0.5 ])
		ct1 = plt.contour(X, Y, knnPreds, lvls, colors = ['k'], label = '')
		ct2 = plt.contour(X, Y, bayesPreds, lvls, colors = ['C1'])
		plt.clabel(ct1, lvls, inline = True, fmt = { 0.5: "KNN"}, fontsize = 12)
		plt.clabel(ct2, lvls, inline = True, fmt = { 0.5: "Bayes"}, fontsize = 12)

		plt.scatter(trainPos0[0], trainPos0[1], marker='o', s = 15, color = 'r', label = '0')
		plt.scatter(trainPos1[0], trainPos1[1], marker='o', s = 15, color = 'b', label = '1')

		plt.xlim([LOW_LIM, HIGH_LIM])
		plt.ylim([LOW_LIM, HIGH_LIM])
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.legend()

		plt.savefig('partD.png')

		print('figure saved')
		print
