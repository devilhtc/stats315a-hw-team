import numpy as np
import util

def main():
	Util = util.Util
	u = Util()
	testUtil(u)

def testUtil(u):
	#print('hello world!')
	c_0, c_1 = u.genCent()
	data_1 = u.genData(c_0, 100)
	data_2 = u.genData(c_1, 100)
	X = np.concatenate((data_1, data_2))
	labels = [0]*100 + [1]*100

	data_1 = u.genData(c_0, 1000)
	data_2 = u.genData(c_1, 1000)
	query = np.concatenate((data_1, data_2))
	queryLabels = [0]*1000 + [1]*1000

	print(u.testLRAccuracy(X, labels, query, queryLabels))

if __name__=='__main__':
	main()
