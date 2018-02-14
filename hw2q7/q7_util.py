import numpy as np

# convert a line (a string of space separated numbers)
# to a list of floats
def line2list(line):
    return [float(x) for x in line.strip().split()]

# read in a txt file as an np array
def readin(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    array_list = [line2list(line) for line in lines if len(line.strip()) > 0]
    f.close()
    return np.array(array_list)

def filter_data(data, keep):
	filtered = []
	for v in data:
		if v[0] in keep:
			filtered.append(v)
	return np.array(filtered)