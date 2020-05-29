import numpy as np


def mse(output, label, derivative=False, total=False):

    if derivative:
        value = output - label
    else:
        value = 0.5 * (output - label) ** 2

    if total:
        value = np.sum(value)
    else:
        pass
    return value


