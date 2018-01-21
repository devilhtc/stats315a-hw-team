import numpy as np
import util

def main():
	Util = util.Util
	u = Util()
	testUtil(u)

def testUtil(u):
	#print('hello world!')
	n_train = 100;
	n_test = 10000;
	c_0, c_1 = u.genCent()
	data_1 = u.genData(c_0, n_train)
	data_2 = u.genData(c_1, n_train)
	X = np.concatenate((data_1, data_2))
	labels = np.asarray([0]*n_train + [1]*n_train)

	data_1 = u.genData(c_0, n_test)
	data_2 = u.genData(c_1, n_test)
	query = np.concatenate((data_1, data_2))
	queryLabels = np.asarray([0]*n_test + [1]*n_test)

	ks = [1, 3, 5, 9, 15, 25, 45, 83, 151]
	u.partB(ks, X, labels, query, queryLabels)

if __name__=='__main__':
	main()
