import numpy as np
import util as u

np.random.seed(315)

data_dir = 'data/'
data_filename = 'spam.data.txt'
indicator_filename = 'spam.traintest.txt'

### tests begin

def test_dividing_train_test():
	X_train, y_train, X_test, y_test = get_spam_Xy_train_test()
	print X_train.shape
	print y_train.shape
	print X_test.shape
	print y_test.shape

def test_indexing():
	A = np.random.randn(3,4)
	idx = np.array([1, 0, 1])
	print A[idx]

def test_preprocess():
	X, y = get_spam_Xy_all(raw = True)
	X2, y = get_spam_Xy_all()

	print X[:3]
	print X2[:3]

def test_readin():
	filename = data_dir + data_filename
	data_array = u.readin(filename)
	X, y = data_array[:, :-1], data_array[:, -1:]
	print data_array.shape
	print X.shape, y.shape

def test_model_1():
	n = 10
	p = 3
	X = np.random.randn(n, p)
	y = np.random.randn(n, 1)
	model = u.SFModel(X, y)
	print y
	print model.y_res

def test_model_2():
	n = 10
	p = 3
	X = np.random.randn(n, p) * 2
	beta = np.random.randn(p, 1) * 3
	y = X.dot(beta) + np.random.randn(n, 1)
	model = u.SFModel(X, y)
	model.build()
	print 'indices (in the order it is selected) are', model.selected_indices
	print 'coefficients at each step is', model.coefficients
	print 'linear regression beta is', u.linear_reg(u.augment(X),y)

def test_import():
	u.test()

### tests end


### helper functions start

# get the whole X and y from file
# if raw, return unprocessed
def get_spam_Xy_all(raw = False):
	filename = data_dir + data_filename
	data_array = u.readin(filename)
	X, y = data_array[:, :-1], data_array[:, -1:]
	if raw:
		return X, y
	X2 = u.preprocess(X)
	return X2, y

# get X_train, y_train, X_test, y_test
# all preprocessed
def get_spam_Xy_train_test():
	X, y = get_spam_Xy_all()
	train_test_indicator = u.readin(data_dir + indicator_filename)
	train_idx, test_idx = u.get_train_test_idx(train_test_indicator)
	X_train, y_train = X[train_idx], y[train_idx]
	X_test, y_test = X[test_idx], y[test_idx]
	return X_train, y_train, X_test, y_test

def main():
	test_dividing_train_test()



if __name__=='__main__':
	main()