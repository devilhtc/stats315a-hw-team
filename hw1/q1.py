import numpy as np
import util

def main():
	Util = util.Util
	u = Util()
	solve(u)

def solve(u):
	#print('hello world!')
	nTrain = 100
	nTest = 10000
	ks = [1, 3, 5, 9, 15, 25, 45, 83, 151]
	f = 10

	centroids = u.genCent()

	X, labels = u.genDataAndLabel(centroids, nTrain)
	query, queryLabels = u.genDataAndLabel(centroids, nTest)

	# part B, test for different k using test on train
	u.partB(ks, X, labels, query, queryLabels)

	# part C, choose optimal k by cross validation
	kOpt = u.partC(X, labels, f, ks)

	print 'optimal k is', kOpt
	# part D, plot scatter of train and decision boundaries for KNN with kOpt and bayes
	u.partD(X, labels, kOpt, centroids)

if __name__=='__main__':
	main()
