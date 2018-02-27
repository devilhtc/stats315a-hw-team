from __future__ import print_function
import numpy as np
import utils as u

# constants
data_dir = 'data/'
train_filename = 'loan_train.csv'
test_filename = 'load_test.csv'

def print_train_values():
	train_lines = u.get_lines(data_dir + train_filename)
	dim_names = u.split_line(train_lines[0])
	dim_ranges, dim_types = u.summarize_range(train_lines[1:])

	train_ranges = zip(dim_names, dim_types, dim_ranges)
	for r in train_ranges:
		print(r)

def main():
	print_train_values()
	print('do we use python?')
	
if __name__ == '__main__':
	main()