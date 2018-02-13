import numpy as np
import time

VERBOSE_DEFAULT = True

# arr is 1d array (s, ), prepend val at the start 
# and return an array of size (s+1, )
def prepend(arr, val):
    return np.array([val] + arr.tolist())

# train and return a model
def train_model(X_train, y_train):
    model = SFModel(X_train, y_train, verbose = False)
    model.build()
    return model

# get X_train, y_train, X_test, y_test
# all preprocessed
def get_spam_Xy_train_test(data_filename, indicator_filename):
    X, y = get_spam_Xy_all(data_filename)
    train_test_indicator = readin(indicator_filename)
    train_idx, test_idx = get_train_test_idx(train_test_indicator)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return X_train, y_train, X_test, y_test

# get the whole X and y from file
# if raw, return unprocessed
def get_spam_Xy_all(filename, raw = False):
    data_array = readin(filename)
    X, y = data_array[:, :-1], data_array[:, -1:]
    if raw:
        return X, y
    X2 = preprocess(X)
    return X2, y

# get the current fold of data based in idx
# return X_train, y_train, X_test, y_test
def get_fold_Xys(X, y, fold_idx):
    n, p = X.shape
    train_idx = np.array([i for i in range(n) if i not in fold_idx])
    test_idx = np.array(fold_idx)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return X_train, y_train, X_test, y_test

# generate indices of f folds of n
# output a list of lists of indices
def generate_fold_idxs(n, f):
    shuffled_idx = np.arange(n)
    np.random.shuffle(shuffled_idx)
    out = [[] for _ in range(f)]
    for i in range(n):
        out[i%f].append(shuffled_idx[i])
    return out

# preprocess X as stated in the problem 
# 1. mark the first 54 dimensions binary zero/nonzero
# 2. use log on the last 3 features
def preprocess(X):
    X2 = X.copy()
    X2[:, :54] = (X2[:, :54] != 0.0)
    X2[:, -3:] = np.log(X2[:, -3:])
    return X2

# get train and test indices based on arr (n, 1)
# with 1 indicating it's test and 0 being train
def get_train_test_idx(train_test_indicator):
    n = len(train_test_indicator)
    train_idx = []
    test_idx = []
    for i in range(n):
        if train_test_indicator[i, 0] > 0:
            test_idx.append(i)
        else:
            train_idx.append(i)
    return np.array(train_idx), np.array(test_idx)
 
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

# remove the mean of each row of X
def col_mean_centered(X):
    return X - np.mean(X, axis = 0, keepdims = True)

# produce the ls fit of x on y
def linear_reg(X, y):
    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)

# orthogonalize x on z (remove the component of x on vector z)
def orthogonalize(x, z):
    return x - x.T.dot(z) / z.T.dot(z) * z

# add a column of 1 to the left of X
def augment(X):
    a, b = X.shape
    return np.concatenate( (np.ones( (a, 1) ), X), axis = 1 )

# step-forward model
class SFModel(object):
    # initialization, only X and y is required
    def __init__(self, X, y, verbose = VERBOSE_DEFAULT):
        n, p = X.shape
        self.n = n
        self.p = p
        self.X = X
        self.y = y
        self.verbose = verbose
        self.setup()

    # setup stepping and outputs
    def setup(self):
        # the current step taken 
        self.step = 0

        # idx (ranging from 0 to p-1) of the dimensions added 
        # (in the order of being added) 
        # and basis
        self.selected_idx = []
        self.zs = [ np.ones(self.n) ]

        # current X_res, y_res (now a 1d vector)
        self.X_res = col_mean_centered( self.X.copy() )
        self.y_res = col_mean_centered( self.y.copy() ) 
        self.y_res = self.y_res.reshape((self.n, ))

        # the coefficients (a list of array with shape (i+1, 1) for the i_th element)
        # the first dimension of each coefficient is the intercept
        self.coefficients = []

    # take the model 1 step forward 
    # return False if already step to the end
    # else return True 
    def one_step_forward(self):
        if self.step >= self.p:
            return False

        unselected_idx = [i for i in range(self.p) if i not in self.selected_idx]
        residues = [] # residues for each index

        for i in unselected_idx:
            # get the ith unselected vector of shape (n,)
            cur_x = self.X_res[:, i].copy()

            # regress self.y_res on cur_x, get the residue after this dimension
            cur_y_res = orthogonalize(self.y_res, cur_x)
            cur_res = np.power( np.linalg.norm(cur_y_res), 2 )
            residues.append((cur_res, i))

        # select the index that gives lowest residue
        selected_i = min(residues)[1]
        selected_column = self.X_res[:, selected_i].copy()

        # remove the component of selected column on self.y_res
        self.y_res = orthogonalize(self.y_res, selected_column)
        self.X_res[:, selected_i] = 0.0 * selected_column # zero out


        if self.verbose:
            print 'At the current step, the residues and idx are'
            print residues
            print 'We then select index =', selected_i
            print 'Residue of y left is now {:0.5f}'.format( np.power(np.linalg.norm(self.y_res), 2) )

        self.zs.append(selected_column)
        self.selected_idx.append(selected_i)

        # orthogonalize rest of self.X_res on the selected column
        for i in unselected_idx:
            if i != selected_i:
                self.X_res[:, i] = orthogonalize(self.X_res[:, i], selected_column)

        self.step += 1
        return True

    def build(self):
        if self.verbose: print 'Build start!'
        start = time.time()
        self.step_forward()
        end = time.time()
        if self.verbose: print 'Build success!'
        if self.verbose: print 'It takes {:0.5f} seconds'.format(end - start)

    # take steps forward to the end
    def step_forward(self):
        while self.one_step_forward():
            if self.verbose: print 'Step', self.step, 'complete\n'
            self.regress_on_current_subset()

    # perform linear regression on the current selected index
    # add coefficient to self.coefficients
    def regress_on_current_subset(self):
        idx_array = np.array(self.selected_idx)

        # get inputs for linear regression of current subset
        cur_X = self.X[:, idx_array]
        cur_X_augmented = augment(cur_X)

        cur_beta = linear_reg(cur_X_augmented, self.y)
        
        # put the beta back to coefficients
        beta_to_add = np.zeros(self.p + 1) 
        beta_to_add[idx_array + 1] += cur_beta[1:, 0]
        beta_to_add[0] = cur_beta[0, 0]

        self.coefficients.append(beta_to_add)

    # get coefficient matrix of shape (p+1, p)
    # where the i_th column is the the coefficient at the i_th step
    # if an argument is given (an index), return 
    def get_coefficient_matrix(self, i = None):
        if i is not None:
            betas = np.stack([self.coefficients[i]])
        else:
            betas = np.stack(self.coefficients)
        return betas.T

    # make a prediction on X0 (m, p) at all steps
    # return a matrix of shape (m, 1) 
    def predict_at_ith_steps(self, X0, i):
        m, p = X0.shape
        assert p == self.p, 'dimensionality mismatch, this models is trained for p = {0} but the input dimension is {1}!'.format(self.p, p)

        # augment X0 and get coefficient matrix
        X0_augmented = augment(X0)
        cm = self.get_coefficient_matrix(i - 1)

        # X0_augmented (m, p+1), cm (p+1, p), y0_hat (m, p)
        y0_hat = X0_augmented.dot(cm)
        return y0_hat

    # make a prediction on X0 (m, p) at all steps
    # return a matrix of shape (m, p) 
    # where the ij_th element is the i_th input predicted at the j_th step
    def predict_at_all_steps(self, X0):
        m, p = X0.shape
        assert p == self.p, 'dimensionality mismatch, this models is trained for p = {0} but the input dimension is {1}!'.format(self.p, p)

        # augment X0 and get coefficient matrix
        X0_augmented = augment(X0)
        cm = self.get_coefficient_matrix()

        # X0_augmented (m, p+1), cm (p+1, p), y0_hat (m, p)
        y0_hat = X0_augmented.dot(cm)
        return y0_hat

