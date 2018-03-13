from __future__ import print_function
import numpy as np
import unittest
import time, datetime
import matplotlib.pyplot as plt
import os

# constants

# inputs
data_dir = 'data/'
train_filename = 'loan_train.csv'
test_filename = 'loan_testx.csv'

# outputs
output_dir = 'outputs/'
output_filename_base = 'loan_testy_'
z = 1.645 # 90 percent interval z score
divide = True

# whether Plot All Features in test
PAF = True

# the range of each non-float names
non_float_range_dict = {
	'employment': ['NA', '< 1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+'],
	'status': ['unchecked', 'partial', 'checked'],
	'reason': ['business', 'cc', 'moving', 'solar', 'home', 'educ', 'renovation', 'boat', 'debt', 'medical', 'event', 'other', 'transport', 'holiday'],
	'quality': ['q1', 'q2', 'q3', 'q4', 'q5', 'q5', 'q6', 'q7'],
	'initial_list_status': ['a', 'b'],
	'term': [' 3 yrs', ' 5 yrs']
}

# plot all the feature distribution in training file
def plot_all_features(filename):
	file_content = get_file_content(filename)
	names = file_content[0][1:]
	types = ['string' if name in non_float_range_dict else 'float' for name in names]
	contents = file_content[1:]
	labels = [float(ele[0]) for ele in contents]
	data = [(names[i], [(c[i + 1] if names[i] in non_float_range_dict else float(c[i + 1])) for c in contents]) for i in range(len(names))]
	num_rows = 5
	num_cols = 6
	f, axarr = plt.subplots(num_rows, num_cols, figsize=(8, 6))
	plt.tight_layout(pad = 0.2, w_pad = 0.2, h_pad = 0.9)
	plt.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05)
	for i in range(len(names)):
		r, c = i / num_cols, i % num_cols
		#print(r, c)
		plot_feature(names[i], types[i], data[i][1], labels, axarr[r][c])
	plt.savefig('all_features.png')

# create bins around the numbers in bins
# input: (n, )
# output: (n+1, )
def expand_bins(bins):
	out = [0] * (len(bins) + 1)
	step = bins[1] - bins[0]
	for i in range(len(bins)):
		out[i] = bins[i] - 0.5 * step 
		out[i + 1] = bins[i] + 0.5 * step 
	return np.array(out)

# generate on a subplot
# values [n]
# labels [n]
def plot_feature(feature_name, feature_type, values, labels, axis):
	# decide which kind of bin to use
	n = len(values)
	if feature_type == 'float':
		# if it is float
		num_bins = 15
		a, b = min(values), max(values)
		bins = np.arange(a, b, (b-a)/num_bins)
		ticks = [('{0:.2f}'.format(bins[i]) if i % 3 == 0 else '') for i in range(len(bins))] 
	else:
		# if string, convert it to numbers
		s = sorted(non_float_range_dict[feature_name])
		ticks = s
		value_dict = {s[i]: i for i in range(len(s))}
		num_bins = len(s)
		bins = np.arange(len(s))
		counts = [0.0] * num_bins
		values = [value_dict[ele] for ele in values]

	# divide labels to 0 and 1
	ebins = expand_bins(bins)
	val0 = [values[i] for i in range(n) if labels[i] == 0.0]
	data0, _ = np.histogram(np.array(val0), ebins)
	val1 = [values[i] for i in range(n) if labels[i] == 1.0]
	data1, _ = np.histogram(np.array(val1), ebins)
	color0 = 'r'
	color1 = 'b'
	scatter_size = 40
	title_fontsize = 9
	tick_fontsize = 7
	scatter_opacity = 0.2
	plot_opacity = 0.7
	#axis.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.03)
	axis.plot(bins, data0, color = color0, alpha = plot_opacity)
	axis.scatter(val0, np.zeros(len(val0)), s = scatter_size, facecolors = 'none', edgecolors=color0, alpha = scatter_opacity)
	axis.plot(bins, data1, color = color1, alpha = plot_opacity)
	axis.scatter(val1, np.zeros(len(val1)), s = scatter_size, facecolors = 'none', edgecolors=color1, alpha = scatter_opacity)
	#axis.set_xticks(bins, ticks)
	axis.tick_params(labelsize = tick_fontsize)
	axis.text(0.5, 1.06, feature_name, horizontalalignment='center', fontsize = title_fontsize, color = '#303030', transform = axis.transAxes)
	#plt.show()

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

def get_file_content(filename):
	lines = get_lines(filename)
	out = [split_line(line) for line in lines]
	return out

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

	def test_plot_feature1(self):
		values = [3.0, 1.0, 1.0, 2.4, 2.2, 4.4]
		labels = map(float,[0, 1, 0, 1, 0, 1])
		#plot_feature('test', 'float', values, labels)

	def test_plot_feature2(self):
		values = ['a', 'b', 'c', 'a', 'b', 'b']
		labels = map(float,[0, 1, 0, 1, 0, 1])
		#plot_feature('test', 'string', values, labels)

	def test_plot_all_features(self):
		data_filename = data_dir + train_filename
		if PAF:
			plot_all_features(data_filename)

def main():
	unittest.main()

if __name__ == '__main__':
	main()
