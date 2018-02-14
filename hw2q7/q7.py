from __future__ import print_function
import numpy as np
import q7_util as u


# constants
test_filename = 'data/zip.test'
train_filename = 'data/zip.train'
keep_digits = [3.0, 5.0, 8.0]

### helper functions (e.g. for reading in data)

def get_data(filename):
	data = u.readin(filename)
	data_arr = u.filter_data(data, keep_digits)
	X = data_arr[:, 1:]
	y = data_arr[:, :1]
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
	data = u.readin(train_filename)
	data_arr = u.filter_data(data, keep_digits)

	# get all data
	X_train, y_train, X_test, y_test = get_all_data()

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
	X_train, y_train, X_test, y_test = get_all_data()
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

def main():
	test_abcde_data()

if __name__=='__main__':
	main()
