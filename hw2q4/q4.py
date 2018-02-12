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
	X = np.random.randn(n, p) * 2
	beta = np.random.randn(p, 1) * 3
	y = X.dot(beta) + np.random.randn(n, 1)
	model = u.SFModel(X, y)
	model.build()
	print 'indices (in the order it is selected) are', model.selected_indices
	print 'coefficients at each step is', model.coefficients
	print 'linear regression beta is', u.linear_reg(u.augment(X),y)

def test_import():
	u.test()

def main():
	test_model_2()

	
if __name__=='__main__':
	main()