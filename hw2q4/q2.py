import numpy as np


def col_mean_centered(x):
    return x - np.mean(x, axis=0, keepdims=True)


def linear_reg(x, y):
    return np.linalg.inv(np.dot(x.T, x)).dot(x.T).dot(y)


def var_rs(x, y, index):
    var = x[:, index]
    beta = linear_reg(var, y)
    res = y - var.dot(beta)
    return res

# orthogonalize a on b
def orthogonalize(x, z):
    return x - x.T.dot(z) / z.T.dot(z) * z


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