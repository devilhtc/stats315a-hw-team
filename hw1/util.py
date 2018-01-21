import numpy as np
import random

class Util():
	def test(self):
		return 'testing complete'
	def gen_cent(self):
		"""
		Generate 10 centroids for each class
		"""
		#mean and conv for class 0
		mean_0 = (1, 0); cov_0 = [[1, 0], [0, 1]]
		c_0 = np.random.multivariate_normal(mean_0, cov_0, (10, ))

		mean_1 = (0, 1); cov_1 = [[1, 0], [0, 1]]
		c_1 = np.random.multivariate_normal(mean_1, cov_1, (10, ))
		return (c_0, c_1)

	def gen_data(self, cen, num_data):
		'''
		#cen are centroids
		#num_data is the number of datasets
		return result (num_data, 2)
		'''
		result = []
		for _ in range(num_data):
			rand_index = random.randint(0, 9)
			mean = cen[rand_index, :]; cov = [[1/5, 0], [0, 1/5]]
			data = np.random.multivariate_normal(mean, cov)
			result.append(data)
		return np.stack(result)
