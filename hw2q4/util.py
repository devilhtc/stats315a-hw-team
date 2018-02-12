import numpy as np
import time

# test import
def test():
    print 'Import success!'

# remove the mean of each row of X
def col_mean_centered(X):
    return X - np.mean(X, axis = 0, keepdims = True)

# produce the ls fit of x on y
def linear_reg(X, y):
    return np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(y)

# orthogonalize x on z (remove the component of x on vector z)
def orthogonalize(x, z):
    return x - x.T.dot(z) / z.T.dot(z) * z

# step-forward model
class SFModel(object):
    # initialization, only X and y is required
    def __init__(self, X, y):
        n, p = X.shape
        self.n = n
        self.p = p
        self.X = X
        self.y = y
        self.setup()

    # setup stepping and outputs
    def setup(self):
        # the current step taken 
        self.step = 0

        # indices (ranging from 0 to p-1) of the dimensions added 
        # (in the order of being added) 
        # and basis
        self.selected_indices = []
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
        print

        if self.step >= self.p:
            return False

        unselected_indices = [i for i in range(self.p) if i not in self.selected_indices]
        residues = [] # residues for each index

        for i in unselected_indices:
            # get the ith unselected vector of shape (n,)
            cur_x = self.X_res[:, i].copy()

            # regress self.y_res on cur_x, get the residue after this dimension
            cur_y_res = orthogonalize(self.y_res, cur_x)
            cur_res = np.linalg.norm(cur_y_res)
            residues.append((cur_res, i))
        print 'At the current step, the residues and indices are'
        print residues
        # select the index that gives 
        selected_index = min(residues)[1]
        selected_column = self.X_res[:, selected_index].copy()
        self.zs.append(selected_column)
        self.selected_indices.append(selected_index)

        # remove the component of selected column on self.y_res
        self.y_res = orthogonalize(self.y_res, selected_column)
        self.X_res[:, selected_index] = 0.0 * selected_column # zero out

        # orthogonalize rest of self.X_res on the selected column
        for i in unselected_indices:
            if i != selected_index:
                self.X_res[:, i] = orthogonalize(self.X_res[:, i], selected_column)

        self.step += 1
        return True

    def build(self):
        print 'Build start!'
        start = time.time()
        self.step_forward()
        end = time.time()
        print 'Build success!'
        print 'It takes {:0.5f} seconds'.format(end - start)

    # take steps forward to the end
    def step_forward(self):
        while self.one_step_forward():
            print 'Step', self.step, 'complete'
            self.regress_on_current_subset()

    # perform linear regression on the current selected index
    # add coefficient to self.coefficients
    def regress_on_current_subset(self):
        pass

    # make a prediction on X0 (m, p) at all steps
    # return a matrix of shape (m, p) 
    # where the ij_th element is the i_th input predicted at the j_th step
    def predict_at_all_steps(self, X0):
        m, p = X0.shape
        assert p==self.p, 'dimensionality mismatch, this models is trained for p = {0} but the input dimension is {1}!'.format(self.p, p)

        # augment X0 and get coefficient matrix
        X0_augmented = np.concatenate( (np.ones(m,1), X0), axis = 1 )
        cm = self.get_coefficient_matrix()

        # X0_augmented (m, p+1), cm (p+1, p), y0_hat (m, p)
        y0_hat = X0_augmented.dot(cm)
        return y0_hat

    # get coefficient matrix of shape (p+1, p)
    # where the i_th column is the the coefficient at the i_th step
    def get_coefficient_matrix(self):
        betas = np.stack(self.coefficients)
        return betas.T


'''
ARCHIVED

# get the residue of y from var
def var_rs(var, y):
    beta = linear_reg(var, y)
    res = y - var.dot(beta)
    return res

def step_forward(x, y):
    (n, p) = x.shape

    x = col_mean_centered(x)
    y = col_mean_centered(y)

    remained_var_index = range(p)
    z_index = []
    se_list = []
    rs = y.copy()
    se = rs.T.dot(rs)

    while len(remained_var_index):
        new_rs = {}
        new_se = {}
        for index in remained_var_index:
            new_rs[index] = var_rs(x, y, index)
            new_se = new_rs[index].T.dot(new_rs[index])

        min_var_index = max(new_se, key=new_se.get)
        remained_var_index.remove(min_var_index)
        z_index.append(min_var_index)
        rs = rs - new_rs[min_var_index]
        se = se - new_se[min_var_index]
        se_list.append(se)
        for index in remained_var_index:
            orthogonalize(x[:, index], min_var_index)

    return z_index, se_list
'''