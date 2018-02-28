from __future__ import print_function
import numpy as np
import utils as u
import models as m

# constants
data_dir = 'data/'
train_filename = 'loan_train.csv'
test_filename = 'loan_testx.csv'


def get_ranges():
	train_range = u.get_file_ranges(data_dir + train_filename)
	test_range = u.get_file_ranges(data_dir + test_filename)
	p1 = len(train_range)
	p2 = len(test_range)

	for i in range(p2):
		print('train', train_range[i+1])
		print('test', test_range[i])
		print()
	print(p1, p2)
	return train_range, test_range

def get_non_float_ranges():
	train_range, test_range = get_ranges()
	p1 = len(train_range)
	p2 = len(test_range)
	lines = []
	for i in range(p2):
		if test_range[i][1] != 'float':
			lines.append((test_range[i][0], test_range[i][2].union(train_range[i+1][2])))
	u.write_lines(lines, data_dir + 'non_float_ranges.txt')

# return X_train (n, p'), Y_train (n, 1), X_test (m, p')
# X_train and X_test are processed in the same way
def get_data():
	train_lines = u.get_lines(data_dir + train_filename)
	names = u.split_line(train_lines[0])
	train_all = u.process_lines_by_name(train_lines[1:], names)
	
	test_lines = u.get_lines(data_dir + test_filename)
	test_all = u.process_lines_by_name(test_lines[1:], names[1:])


	train_XY = np.array(train_all)
	X_train = train_XY[:, 1:]
	Y_train = train_XY[:, :1]
	X_test = np.array(test_all)

	print('X_train shape', X_train.shape)
	print('Y_train shape', Y_train.shape)
	print('X_test shape', X_test.shape)
	return X_train, Y_train, X_test


def main():
	get_data()
	#print('do we use python?')
	
if __name__ == '__main__':
	main()