import numpy as np
from dsk.costs.cost_functions import mse


def r_squared(y, f_fitted):

    ss_mean = mse(np.array([np.mean(y) for _ in range(y.shape[1])]), y, total=True)
    ss_fit = mse(f_fitted, y, total=True)
    return (ss_mean - ss_fit) / ss_mean
