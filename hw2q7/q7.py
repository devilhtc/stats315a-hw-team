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


def main():

	get_train_data()

if __name__=='__main__':
	main()
	