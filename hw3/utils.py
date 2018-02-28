from __future__ import print_function
import numpy as np
import unittest


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

def main():
	unittest.main()

if __name__ == '__main__':
	main()
