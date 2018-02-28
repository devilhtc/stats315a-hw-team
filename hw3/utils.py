from __future__ import print_function
import numpy as np
import unittest


# process a value by its name, returns a list of values
# e.g. name: age, 18 -> [18],  or name: reason, 'travel': [0, 1, 0, 0, 0, 0]
def process_val_by_name(val, name):
	non_float_range_dict = {
		'employment': ['NA', '< 1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+'],
		'status': ['unchecked', 'partial', 'checked'],
		'reason': ['business', 'cc', 'moving', 'solar', 'home', 'educ', 'renovation', 'boat', 'debt', 'medical', 'event', 'other', 'transport', 'holiday'],
		'quality': ['q1', 'q2', 'q3', 'q4', 'q5', 'q5', 'q6', 'q7'],
		'initial_list_status': ['a', 'b'],
		'term': [' 3 yrs', ' 5 yrs']
	}

	# convert a name to a one-hot vector with 1 at its place
	def one_hot(name, name_list):
		return [(1.0 if ele == name else 0.0) for ele in name_list]

	# defines how we process the data
	process_dict = {
		'initial_list_status': lambda x: [1.0] if x == 'a' else [0.0],
		'term': lambda x: [1.0] if '5' in x else [0.0],
		'quality': lambda x: [float(x[1:])],
		'employment': lambda x: one_hot(x, non_float_range_dict['employment']),
		'reason': lambda x: one_hot(x, non_float_range_dict['reason']),
		'status': lambda x: one_hot(x, non_float_range_dict['status']),
	}

	def default_process_func(x):
		return [float(x)]

	process_func = process_dict.get(name, default_process_func)
	return process_func(val)

# process the lines by its identifying names
# return a list of list of floats for analysis
def process_lines_by_name(lines, names):
	processed_lines = []
	for line in lines:
		cur_line = []
		splitted_line = split_line(line)
		for i in range(len(names)):
			cur_line = cur_line + process_val_by_name(splitted_line[i], names[i])
		processed_lines.append(cur_line)
	return processed_lines

# get range of data stored in the file
def get_file_ranges(filename):
	lines = get_lines(filename)
	dim_names = split_line(lines[0])
	dim_ranges, dim_types = summarize_range(lines[1:])
	train_ranges = zip(dim_names, dim_types, dim_ranges)
	return train_ranges

# split a line by ',' then strip out the '"'s
def split_line(line):
	return [ele.strip('"') for ele in line.split(',')]

# test if a string is a float
def isfloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

# get stripped lines from a file 
def get_lines(filename):
	f = open(filename, 'r')
	lines = []
	for line in f.readlines():
		lines.append(line.strip())
	f.close()
	return lines

def write_lines(lines, filename):
	f = open(filename, 'w+')
	for line in lines:
		f.write(str(line) + '\n')
	f.close()

# return 'float' if all the strings can be convereted to float, otherwise return 'string'
def get_types(set_of_values):
	return 'float' if all(isfloat(val) for val in set_of_values) else 'string'

# summarize the range of the lines
# input: lines [str]
# output: values [set()]
def summarize_range(lines):
	if len(lines) == 0:
		return []
	p = len(lines[0].split(','))
	values = [set([]) for _ in range(p)]
	for line in lines:
		line_vals = split_line(line)
		for i in range(p):
			values[i].add(line_vals[i])
	value_types = [get_types(vals) for vals in values]
	for i in range(p):
		if value_types[i] == 'float':
			values[i] = set([float(val) for val in values[i]])
			values[i] = set(['min:{0:.2f}'.format(min(values[i])), 'max:{0:.2f}'.format(max(values[i]))])
	return values, value_types

class UtilTests(unittest.TestCase):
	def setUp(self):
		self.num = 0
		self.s_lines = ['1,2,"blue"', '2,3,"red"']
		self.s_values = [set(['min:1.00', 'max:2.00']), set(['min:2.00', 'max:3.00']), set(["blue", "red"])]
		self.s_types = ['float', 'float', 'string']

	def test_dummy(self):
		self.assertEqual(self.num, 0, 'dummy test failed')

	def test_summarize(self):
		vs, ts = summarize_range(self.s_lines)
		self.assertEqual(vs, self.s_values, 'summarize_range values failed')
		self.assertEqual(ts, self.s_types, 'summarize_range types failed')

	def test_process(self):
		pass

def main():
	unittest.main()

if __name__ == '__main__':
	main()
