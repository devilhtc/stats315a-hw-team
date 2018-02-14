import numpy as np

# get top k principal components
def get_topk_pc(data, k):
	U, D, V = np.linalg.svd(data)
	V = V.T
	return V[:,:k]

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

# data is of dimension n * p
# p = l * l
# filter it to p' = (l/kernel_size) * (l/kernel_size)
def ave_pool(data, kernel_size = 4):
	k = kernel_size
	n, p = data.shape
	l = int(np.sqrt(p))
	pp = l/k

	data_reshaped = data.reshape((n, l, l))
	output = np.zeros((n, pp, pp))
	for t in range(n):
		for i in range(pp):
			for j in range(pp):
				output[t, i, j] += np.mean(data_reshaped[t, i * k : i * k + k, j * k : j * k + k])

	return output.reshape((n, -1))

# data is of dimension n * p
# p = l * l
# filter it to p' = (l/kernel_size) * (l/kernel_size)
def ave_pool2(data, kernel_size = 4):
	k = kernel_size
	n, p = data.shape
	l = int(np.sqrt(p))
	pp = l/k

	data_reshaped = data.reshape((n, l, l))
	output = np.zeros((n, l, l))
	for t in range(n):
		for i in range(pp):
			for j in range(pp):
				output[t, i * k : i * k + k, j * k : j * k + k] += np.mean(data_reshaped[t, i * k : i * k + k, j * k : j * k + k])

	return output.reshape((n, -1))


