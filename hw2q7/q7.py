import numpy as np
import q7_util as u


# constants
test_filename = 'data/zip.test'
train_filename = 'data/zip.train'
keep_digits = [3.0, 5.0, 8.0]

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

def examples():
	X_train, y_train, X_test, y_test = get_all_data()
	print 'train', X_train.shape, y_train.shape
	print 'test', X_test.shape, y_test.shape

	X_train_pooled = u.ave_pool(X_train)
	print X_train_pooled.shape
	print X_train_pooled[:5]

def main():
	examples()

if __name__=='__main__':
	main()
