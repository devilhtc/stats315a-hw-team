import numpy as np
import util as u

np.random.seed(315)

data_dir = 'data/'
data_filename = 'spam.data.txt'
indicator_filename = 'spam.traintest.txt'
num_folds = 10

### tests begin
def test_folds():
	X, y, _, _ = get_all_data()
	n, p = X.shape
	print X.shape
	fold_idxs = u.generate_fold_idxs(n, num_folds)
	for fold_idx in fold_idxs:
		X_train, y_train, X_test, y_test = u.get_fold_Xys(X, y, fold_idx)
		print 
		print 'current fold indices have length', len(fold_idx)
		print 'training set shapes:', X_train.shape, y_train.shape
		print 'test set shapes:', X_test.shape, y_test.shape

def test_dividing_train_test():
	X_train, y_train, X_test, y_test = get_all_data()
	print X_train.shape
	print y_train.shape
	print X_test.shape
	print y_test.shape

def test_indexing():
	A = np.random.randn(3,4)
	idx = np.array([1, 0, 1])
	print A[idx]

def test_preprocess():
	filename = data_dir + data_filename
	X, y = u.get_spam_Xy_all(filename, raw = True)
	X2, y = u.get_spam_Xy_all(filename)

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
	# generate some random data
	X = np.random.randn(n, p) * 2
	beta = np.random.randn(p, 1) * 3
	y = X.dot(beta) + np.random.randn(n, 1)

	# try build a model on it
	model = u.SFModel(X, y)
	model.build()
	print 'indices (in the order it is selected) are', model.selected_indices
	print 'coefficients at each step is', model.coefficients
	print 'linear regression beta is', u.linear_reg(u.augment(X),y)

def test_import():
	u.test()

### tests end

### helper functions start
def get_all_data():
	f1 = data_dir + data_filename # data filename
	f2 = data_dir + indicator_filename # indicator filename
	return u.get_spam_Xy_train_test(f1, f2)

### helper functions end

'''                     '''
'''      hw begins!     ''' 
'''                     '''

def main():
	test_folds()



if __name__=='__main__':
	main()