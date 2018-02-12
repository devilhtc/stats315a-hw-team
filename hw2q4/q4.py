import numpy as np
import util as u

np.random.seed(315)

def test_model_1():
	n = 10
	p = 3
	X = np.random.randn(n, p)
	y = np.random.randn(n, 1)
	model = u.SFModel(X, y)
	print y
	print model.y_res

def test_model_2():
	n = 10
	p = 3
	X = np.random.randn(n, p)
	y = np.random.randn(n, 1)
	model = u.SFModel(X, y)
	model.build()
	print model.selected_indices

def test_import():
	u.test()

def main():
	test_model_2()

	
if __name__=='__main__':
	main()