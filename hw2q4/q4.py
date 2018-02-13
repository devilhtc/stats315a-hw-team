import numpy as np
import matplotlib.pyplot as plt
import q4_util as u

np.random.seed(315)

# constants
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

'''                      '''
'''                      '''
'''       HW BEGINS      ''' 
'''                      '''
'''                      '''

def partF():

	print 'part F'

	print 'getting data...'
	X, y, _, _ = get_all_data()
	n, p = X.shape
	steps = np.arange(p) + 1

	# train and test model
	print 'training and testing model...'
	model = u.train_model(X, y)
	y_hat = model.predict_at_all_steps(X)
	y_hat_binary = (y_hat > 0.5)
	y_augmented = np.repeat(y, p, axis = 1)

	print 'making plots...'

	# 1. plot rss vs step
	rss = np.sum(np.power(y_hat - y_augmented, 2), axis = 0)
	plt.figure('f1')
	plt.plot(steps, rss)
	plt.xlabel('step')
	plt.ylabel('rss')
	plt.savefig('partF1.png')

	# 2. plot misclassification error vs step
	err = np.sum( np.abs(y_hat_binary - y_augmented), axis = 0) / float(n)
	plt.figure('f2')
	plt.plot(steps, err)
	plt.xlabel('step')
	plt.ylabel('misclassification error')
	plt.savefig('partF2.png')

	print 'saved'
	print 


def partG():
	print 'part G'
	print 'getting data and dividing into folds...'

	# get data and fold indices
	X, y, _, _ = get_all_data()
	n, p = X.shape
	steps = np.arange(p) + 1
	fold_idxs = u.generate_fold_idxs(n, num_folds)

	# aggregate error at each fold
	fold_errors = []

	print 'training and testing on the folds...'
	for i in range(num_folds):
		fold_idx = fold_idxs[i]
		# get train and test for this fold
		X_train, y_train, X_test, y_test = u.get_fold_Xys(X, y, fold_idx)
		n_test = X_test.shape[0]
		# predict using the trained model
		fold_model = u.train_model(X_train, y_train)
		y_test_hat = fold_model.predict_at_all_steps(X_test)
		y_test_hat_binary = y_test_hat > 0.5

		# produce error at each step
		y_augmented = np.repeat(y_test, p, axis = 1)
		err_arr = np.mean( np.abs(y_test_hat_binary - y_augmented), axis = 0)
		fold_errors.append(err_arr)

		print 'fold {0} done'.format(i + 2)

	print 'analyzing data and making plots...'

	# get the mean and std for all folds at each step
	fold_errors_array = np.array(fold_errors)
	error_mean = np.mean(fold_errors_array, axis = 0)
	error_std = np.std(fold_errors_array, axis = 0) / np.sqrt(num_folds)

	plt.figure('g')
	plt.plot(steps, error_mean, label = 'mean')
	plt.plot(steps, error_mean + error_std, color = 'C1', linestyle = '--', label = '1 std interval')
	plt.plot(steps, error_mean - error_std, color = 'C1', linestyle = '--')
	plt.xlabel('step')
	plt.ylabel('misclassification error')
	plt.legend()
	plt.savefig('partG.png')	

	print 'saved'
	print

def partH():
	print 'part H'
	print 'getting data...'

	# with 28 chosen 
	chosen = 28
	print 'we chose to take {0} steps based on previous results'.format(chosen)

	X_train, y_train, X_test, y_test = get_all_data()
	n, p = X_train.shape
	steps = np.arange(p) + 1

	print 'training and testing model...'
	model = u.train_model(X_train, y_train)
	y_test_hat = model.predict_at_all_steps(X_test)
	y_test_hat_binary = y_test_hat > 0.5

	y_augmented = np.repeat(y_test, p, axis = 1)
	err_arr = np.mean( np.abs(y_test_hat_binary - y_augmented), axis = 0)

	print 'making plot...'

	plt.figure('h')
	plt.plot(steps, err_arr, label = 'test error')
	plt.scatter([chosen - 1], [err_arr[chosen - 1]], marker = 'o', color = 'C1', s = 100, label = 'chosen step')
	plt.xlabel('step')
	plt.ylabel('misclassification error')
	plt.legend()
	plt.savefig('partH.png')

	print 'saved'
	print

def partI():
	print 'part I'
	print 'getting data...'
	X_train, y_train, X_test, y_test = get_all_data()
	n, p = X_train.shape
	steps = np.arange(p) + 1

	print 'training model and extract coefficients...'
	model = u.train_model(X_train, y_train)
	
	print 'analyzing data...'
	# get coefficients
	cm = model.get_coefficient_matrix()
	selected_idx = model.selected_idx
	intercepts = u.prepend(cm[0, :], np.mean(y_test))
	intercepts_steps = np.arange(p + 1)

	# calculate R squared
	y_train_hat = model.predict_at_all_steps(X_train)
	y_train_augmented = np.repeat(y_train, p, axis = 1)
	ss_res = np.sum(np.power(y_train_hat - y_train_augmented, 2), axis = 0)
	ss_tot = np.sum( np.power(y_train - np.mean(y_train), 2) )
	R_squared = 1.0 - ss_res / ss_tot
	R_squared_with_null = u.prepend(R_squared, 0.0)

	print 'making plots...'

	# plot the coefficients path vs step
	plt.figure('i1')
	plt.xlabel('step')
	plt.ylabel('coefficient')
	plt.plot(intercepts_steps, intercepts, label = 'intercept')
	for i in range(p):
		idx = selected_idx[i]
		if i < 10:
			plt.plot(intercepts_steps, u.prepend(cm[idx+1, :], 0.0), label = str(idx + 1))
		else:
			plt.plot(intercepts_steps, u.prepend(cm[idx+1, :], 0.0))
	plt.legend()
	plt.savefig('partI1.png')

	# plot the coefficients path vs R squared
	plt.figure('i2')
	plt.xlabel('R^2')
	plt.ylabel('coefficient')
	plt.plot(R_squared_with_null, intercepts, label = 'intercept')
	for i in range(p):
		idx = selected_idx[i]
		if i < 10:
			plt.plot(R_squared_with_null, u.prepend(cm[idx+1, :], 0.0), label = str(idx + 1))
		else:
			plt.plot(R_squared_with_null, u.prepend(cm[idx+1, :], 0.0))
	plt.legend()
	plt.savefig('partI2.png')

	print 'saved'
	print 	

def main():
	#test_folds()
	partF()
	partG()
	partH()
	partI()

if __name__=='__main__':
	main()