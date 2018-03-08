from __future__ import print_function
import numpy as np
import unittest
import datetime
import os

# constants
output_dir = 'outputs/'
output_filename_base = 'loan_testy_'
z = 1.645 # 90 percent interval z score
divide = True

# input is all the default rates as an np array
# output file in outputs/ with time stamp
def output_to_file(default_rates):
	default_rates = np.array(default_rates).flatten()
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	a, b = confidence_interval(default_rates)
	out_filename = output_dir + output_filename_base + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + '.csv'
	print('output file {0}'.format(out_filename))
	f = open(out_filename, 'w+')
	f.write('{0}, {1}\n'.format(a, b))
	f.write('\n'.join(map(str, default_rates.tolist())))
	f.close()

# calculate the confidence interval of default rates
# return a, b
def confidence_interval(default_rates):
	sigma = np.std(default_rates)
	mu = np.mean(default_rates)
	n = len(default_rates) if divide else 1
	return mu - z * sigma / np.sqrt(n), mu + z * sigma / np.sqrt(n)

# using the index of the current test set, get the the train and test data for this fold
def get_fold_XYs(X, Y, fold_idx):
    n, p = X.shape
    train_idx = np.array([i for i in range(n) if i not in fold_idx])
    test_idx = np.array(fold_idx)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    return (X_train, Y_train, X_test, Y_test)

# generate the folds, takes inputs X, Y and f (number of folds)
# return a list of tuples in the form of (X_train, Y_train, X_test, Y_test)
def generate_folds(X, Y, f, seed = None):
	if seed:
		np.random.seed(seed)
	n, _ = X.shape
	fold_idxs = generate_fold_idxs(n, f)
	out = []
	for fold_idx in fold_idxs:
		out.append(get_fold_XYs(X, Y, fold_idx))
	return out

# generate indices of f folds of n
# output a list of lists of indices
def generate_fold_idxs(n, f):
    shuffled_idx = np.arange(n)
    np.random.shuffle(shuffled_idx)
    out = [[] for _ in range(f)]
    for i in range(n):
        out[i%f].append(shuffled_idx[i])
    return out

# add a column of 1 to the left of X
def augment(X):
    a, b = X.shape
    return np.concatenate( (np.ones( (a, 1) ), X), axis = 1 )

# produce the ls fit of x on y
def linear_reg(X, y):
    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)

# convert a name to a one-hot vector with 1 at its place
def one_hot(name, name_list):
	return [(1.0 if ele == name else 0.0) for ele in name_list]

def one_hot_range(name, range_list):
	return [(1.0 if name in ele else 0.0) for ele in range_list]

# process a value by its name, returns a list of values
# e.g. name: age, 18 -> [18],  or name: reason, 'travel': [0, 1, 0, 0, 0, 0]
def process_val_by_name(val, name, method = 1):
	# the range of each non-float names
	non_float_range_dict = {
		'employment': ['NA', '< 1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+'],
		'status': ['unchecked', 'partial', 'checked'],
		'reason': ['business', 'cc', 'moving', 'solar', 'home', 'educ', 'renovation', 'boat', 'debt', 'medical', 'event', 'other', 'transport', 'holiday'],
		'quality': ['q1', 'q2', 'q3', 'q4', 'q5', 'q5', 'q6', 'q7'],
		'initial_list_status': ['a', 'b'],
		'term': [' 3 yrs', ' 5 yrs']
	}

	# process_dict1/2/3 ... defines how we process the data

	# binary on initial_list_status and term
	# converts quality to float
	# one-hot on the rest
	process_dict1 = {
		'initial_list_status': lambda x: [1.0] if x == 'a' else [0.0],
		'term': lambda x: [1.0] if '5' in x else [0.0],
		'quality': lambda x: [float(x[1:])],
		'employment': lambda x: one_hot(x, non_float_range_dict['employment']),
		'reason': lambda x: one_hot(x, non_float_range_dict['reason']),
		'status': lambda x: one_hot(x, non_float_range_dict['status']),
	}

	employment_range_list = [['NA', '< 1', '1'], ['2', '3', '4'], ['5', '6', '7'], ['8', '9', '10+']]
	quality_range_list = [['q1', 'q2', 'q3'], ['q4', 'q5'], ['q6', 'q7']]

	# segment employment to four segments (changed from dict1)
	process_dict2 = dict(process_dict1)
	process_dict2['employment'] = lambda x: one_hot_range(x, employment_range_list)

	# drops reason from dict1
	process_dict3 = dict(process_dict1)
	process_dict3['reason'] = lambda x: []

	# segment quality to three segments (changed from dict1)
	process_dict4 = dict(process_dict1)
	process_dict4['quality'] = lambda x: one_hot_range(x, quality_range_list)

	# segment employment to four segments (changed from dict1) 
	# and dropped reason
	process_dict5 = dict(process_dict1)
	process_dict5['employment'] = lambda x: one_hot_range(x, employment_range_list)
	process_dict5['reason'] = lambda x: []

	# segment quality to three segments (changed from dict1)
	# and dropped reason
	process_dict6 = dict(process_dict1)
	process_dict6['quality'] = lambda x: one_hot_range(x, quality_range_list)
	process_dict6['reason'] = lambda x: []	

	# segment quality to three segments (changed from dict1)
	# segment employment to four segments (changed from dict1) 
	process_dict7 = dict(process_dict1)
	process_dict7['employment'] = lambda x: one_hot_range(x, employment_range_list)
	process_dict7['quality'] = lambda x: one_hot_range(x, quality_range_list)

	# segment quality to three segments (changed from dict1)
	# segment employment to four segments (changed from dict1) 
	# and dropped reason
	process_dict8 = dict(process_dict1)
	process_dict8['employment'] = lambda x: one_hot_range(x, employment_range_list)
	process_dict8['quality'] = lambda x: one_hot_range(x, quality_range_list)
	process_dict8['reason'] = lambda x: []	

	process_dicts = {
		1: process_dict1,
		2: process_dict2,
		3: process_dict3,
		4: process_dict4,
		5: process_dict5,
		6: process_dict6,
		7: process_dict7,
		8: process_dict8
	}

	# default processing function, 
	def default_process_func(x):
		return [float(x)]

	process_dict = process_dicts[method]
	process_func = process_dict.get(name, default_process_func)
	return process_func(val)

# process the lines by its identifying names
# return a list of list of floats for analysis
def process_lines_by_name(lines, names, method = 1):
	processed_lines = []
	for line in lines:
		cur_line = []
		splitted_line = split_line(line)
		for i in range(len(names)):
			cur_line = cur_line + process_val_by_name(splitted_line[i], names[i], method = method)
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

	def test_one_hot_range(self):
		employment_range_list = [['NA', '< 1', '1'], ['2', '3', '4'], ['5', '6', '7'], ['8', '9', '10+']]
		self.assertEqual(one_hot_range('1', employment_range_list), [1, 0, 0, 0], 'one_hot_range test 1 failed')
		self.assertEqual(one_hot_range('9', employment_range_list), [0, 0, 0, 1], 'one_hot_range test 2 failed')

	def test_write(self):
		rates = np.arange(0.0, 1.0, 0.01)
		output_to_file(rates)

def main():
	unittest.main()

if __name__ == '__main__':
	main()
