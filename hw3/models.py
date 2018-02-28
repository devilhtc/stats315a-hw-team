import numpy as np
import sklearn.linear_model as sklm
import unittest
import utils as u

'''
models.py

file for statistic learning models

each model 
1. should implement the object BasicModel, it should have at least 3 more methods: fit, predict and get_params
2. should be initiazed with a set a parameters 
3. should be given X (n, p), Y (n, 1) when fitting
4. should be given X (n, p) when testing
5. should return a dictionary mapping name of the parameter to its value
	(returns empty dictionary if not possible)
	e.g. {'beta': np.array([[-1.0, 2.0, 3.0]])}
'''


# basic model for the other models to implement
class BasicModel(object):
	def __init__(self):
		raise NotImplementedError('each model should implement it')

	def fit(self, X, Y):
		raise NotImplementedError('each model should implement it')

	def predict(self, X):
		raise NotImplementedError('each model should implement it')

	def get_params(self):
		raise NotImplementedError('each model should implement it')

# just for testing - sklearn linear model, sklearn models takes in the same dimension as our model
class SKLinearModel(BasicModel):
	def __init__(self):
		self.model = sklm.LinearRegression()
		self.fitted = False
		self.params = {}

	def fit(self, X, Y):
		self.model.fit(X, Y)
		self.fitted = True
		beta = np.concatenate([self.model.intercept_, self.model.coef_.flatten()], axis = 0).reshape((-1, 1))
		self.params['beta'] = beta
		
	def predict(self, X):
		if not self.fitted:
			raise Exception('This model was not yet fitted')
		return self.model.predict(X)

	def get_params(self):
		if not self.fitted:
			raise Exception('This model was not yet fitted')
		return self.params

# a linear model using (XTX)^(-1)XTY as beta
class LinearModel(BasicModel):
	def __init__(self):
		self.fitted = False
		self.beta = None 
		self.params = {}

	def fit(self, X, Y):
		self.beta = u.linear_reg(u.augment(X), Y)
		self.fitted = True
		self.params['beta'] = self.beta

	def predict(self, X):
		if not self.fitted:
			raise Exception('This model was not yet fitted')
		return u.augment(X).dot(self.beta)

	def get_params(self):
		if not self.fitted:
			raise Exception('This model was not yet fitted')
		return self.params

# ridge regression model
class RidgeModel(BasicModel):
	def __init__(self, alpha = 1.0):
		self.model = sklm.Ridge(alpha = alpha)
		self.fitted = False
		self.params = {'alpha': alpha}

	def fit(self, X, Y):
		self.model.fit(X, Y)
		self.fitted = True
		beta = np.concatenate([self.model.intercept_, self.model.coef_.flatten()], axis = 0).reshape((-1, 1))
		self.params['beta'] = beta
		
	def predict(self, X):
		if not self.fitted:
			raise Exception('This model was not yet fitted')
		return self.model.predict(X)

	def get_params(self):
		if not self.fitted:
			raise Exception('This model was not yet fitted')
		return self.params


# test the models
class TestModels(unittest.TestCase):
	def setUp(self):
		# a linear model of beta = [-1, 2, 3] should fit it perfectly
		self.X_train = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
		self.Y_train = np.array([[1.0], [2.0], [-1.0]])
		self.X_test = np.array([ [1.0, 1.0], [1.5, 1.0]])
		
	def test_not_fitted(self):
		lm = SKLinearModel()
		try:
			lm.predict(self.X_test)
			self.assertEqual(0, 1, msg = 'SKLinearModel able to predict without fitting')
		except:
			self.assertEqual(0, 0, msg = 'Allisgud')
			
	def test_basic_model(self):
		try:
			b = BasicModel()
			self.assertEqual(0, 1, msg = 'BasicModel was initiazed (which should never happend)')
		except NotImplementedError:
			self.assertEqual(0, 0, msg = 'Allisgud')

	def test_sklinear_model(self):
		X_train = self.X_train
		Y_train = self.Y_train
		X_test = self.X_test
		
		lm = SKLinearModel()
		lm.fit(X_train, Y_train)

		Y_test = lm.predict(X_test)

		X_train_augmented = u.augment(X_train)
		beta = u.linear_reg(X_train_augmented, Y_train)
		Y_test_true = u.augment(X_test).dot(beta)
		
		m = Y_test.shape[0]
		for i in range(m):
			self.assertEqual(int(Y_test[i, 0]), int(Y_test_true[i, 0]), msg = 'SKLinearModel coefficients wrong')

	def test_linear_model(self):
		lm = LinearModel()
		lm.fit(self.X_train, self.Y_train)
		Y_test = lm.predict(self.X_test[:1, :])
		self.assertAlmostEqual(Y_test[0, 0], 4.0, msg = 'LinearModel predictions wrong')

	def test_ridge_model(self):
		rm = RidgeModel(alpha = 1.0)
		rm.fit(self.X_train, self.Y_train)
		beta = rm.get_params()['beta']
		true_beta = np.array([0.25, 0.375, 0.875]).reshape((-1, 1))
		p = beta.shape[0]
		for i in range(p):
			self.assertAlmostEqual(beta[i, 0], true_beta[i, 0], msg = 'RidgeModel coefficients wrong')

def main():
	unittest.main()

if __name__ == '__main__':
	main()
