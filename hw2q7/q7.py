from __future__ import print_function
import numpy as np
import q7_util as u
import glmnet_python
from glmnet import glmnet
from glmnetPredict import glmnetPredict
from glmnetPlot import glmnetPlot
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy

# constants
test_filename = 'data/zip.test'
train_filename = 'data/zip.train'
keep_digits = [3.0, 5.0, 8.0]
alpha_step = 0.1

### helper functions (e.g. for reading in data)

def get_data(filename):
	data = u.readin(filename)
	data_arr = u.filter_data(data, keep_digits)
	X = np.array(data_arr[:, 1:], dtype="float64")
	y = np.array(data_arr[:, :1], dtype="float64")
	return X, y

def get_all_data():
	X_train, y_train = get_data(train_filename)
	X_test, y_test = get_data(test_filename)
	return X_train, y_train, X_test, y_test

def get_all_data_a():
	return get_all_data()

def get_all_data_b():
	X_train, y_train, X_test, y_test = get_all_data()
	k = 30
	topk_pc = u.get_topk_pc(X_train, k)
	X_train_mapped = X_train.dot(topk_pc)
	X_test_mapped = X_test.dot(topk_pc)

	return X_train_mapped, y_train, X_test_mapped, y_test

def get_all_data_c():
	# get training data (for svd)
	data_all = u.readin(train_filename)
	data_kept = u.filter_data(data_all, keep_digits)
	data_train_mean = np.mean(data_kept[:, 1:], axis = 0)
	data_kept[:, 1:] -= data_train_mean
	data_arr = data_kept
	# get all data

	X_train, y_train, X_test, y_test = get_all_data()
	X_train -= data_train_mean
	X_test -= data_train_mean

	k = 10
	pcs = []

	for d in keep_digits:
		# get the data with current digit
		cur_data = u.filter_data(data_arr, [d])

		# get top k components
		cur_topk_pc = u.get_topk_pc(cur_data[:, 1:], k)
		pcs.append(cur_topk_pc)

	all_pc = np.concatenate(pcs, axis = 1)
	X_train_mapped = X_train.dot(all_pc)
	X_test_mapped = X_test.dot(all_pc)

	return X_train_mapped, y_train, X_test_mapped, y_test

def get_all_data_d():
	X_train, y_train, X_test, y_test = get_all_data()

	X_train_pooled = u.ave_pool(X_train)
	X_test_pooled = u.ave_pool(X_test)
	return X_train_pooled, y_train, X_test_pooled, y_test

def get_all_data_e():
	return get_all_data_d()

### helper functions

def test_abcde_data():
	print( 'parta' )
	X_train, y_train, X_test, y_test = get_all_data_a()
	print( 'train', X_train.shape, y_train.shape )
	print( 'test', X_test.shape, y_test.shape )
	print(  )

	print( 'partb' )
	X_train, y_train, X_test, y_test = get_all_data_b()
	print( 'train', X_train.shape, y_train.shape ) 
	print( 'test', X_test.shape, y_test.shape )
	print(  )

	print( 'partc' )
	X_train, y_train, X_test, y_test = get_all_data_c()
	print( 'train', X_train.shape, y_train.shape )
	print( 'test', X_test.shape, y_test.shape )
	print(  )

	print( 'partd' )
	X_train, y_train, X_test, y_test = get_all_data_d()
	print( 'train', X_train.shape, y_train.shape )
	print( 'test', X_test.shape, y_test.shape )
	print(  )

	print( 'parte' )
	X_train, y_train, X_test, y_test = get_all_data_e()
	print( 'train', X_train.shape, y_train.shape )
	print( 'test', X_test.shape, y_test.shape )
	print(  )

def pred_error(pred, y):
	y = y.reshape(-1)
	return np.mean(pred != y)

def test_e():
	X_train, y_train, X_test, y_test = get_all_data_e()
	alpha_values = np.arange(0, 1.01, alpha_step)
	for alpha in alpha_values:
		fit = glmnet(x = X_train.copy(), y = y_train.copy(), family = 'multinomial', alpha = alpha)
		lambdau = fit['lambdau']
		devs = fit['dev']
		fc = glmnetPredict(fit, X_test, ptype = 'class', s = lambdau).reshape(lambdau.shape[0], -1).T
		errs = [pred_error(fc[:, i], y_test) for i in range(fc.shape[-1])]
		plt.plot(devs, errs, label="alpha=" + str(alpha))
		plt.legend()
	plt.xlabel("Deviance Ratio")
	plt.ylabel("Prediction Error")
	plt.show()

def e_error():
	print("e:")
	X_train, y_train, X_test, y_test = get_all_data_e()
	fit = glmnet(x = X_train.copy(), y = y_train.copy(), family = 'multinomial')
	fc = glmnetPredict(fit, X_train, ptype='class', s=scipy.array([0.0])).T
	train_err = pred_error(fc, y_train)
	print("train error:" + str(train_err))

	fc = glmnetPredict(fit, X_test, ptype = 'class', s = scipy.array([0.0])).T
	test_err = pred_error(fc, y_test)
	print("test error:" + str(test_err))

def lda(X_train, y_train, X_test, y_test):
	clf = LinearDiscriminantAnalysis()
	clf.fit(X_train, y_train)
	LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
							   solver='svd', store_covariance=False, tol=0.0001)
	train_pred = clf.predict(X_train)
	train_error = pred_error(train_pred, y_train.reshape(-1))
	print("train error:" + str(train_error))

	pred = clf.predict(X_test)
	test_error = pred_error(pred, y_test.reshape(-1))
	print("test error:" + str(test_error))

def lda_error():
	print("a:")
	X_train, y_train, X_test, y_test = get_all_data_a()
	lda(X_train, y_train.reshape(-1), X_test, y_test.reshape(-1))
	print("b:")
	X_train, y_train, X_test, y_test = get_all_data_b()
	lda(X_train, y_train.reshape(-1), X_test, y_test.reshape(-1))
	print("c:")
	X_train, y_train, X_test, y_test = get_all_data_c()
	lda(X_train, y_train.reshape(-1), X_test, y_test.reshape(-1))
	print("d:")
	X_train, y_train, X_test, y_test = get_all_data_d()
	lda(X_train, y_train.reshape(-1), X_test, y_test.reshape(-1))

def main():
	lda_error()
	e_error()

if __name__=='__main__':
	main()
