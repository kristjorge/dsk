import numpy as np


def mse(output, label, derivative=False, total=False):

    if not derivative:
        value = 0.5 * (output - label) ** 2
        if not total:
            return value
        else:
            value = np.sum(value)
        return value

    else:
        return output - label

