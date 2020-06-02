import math as m
import numpy as np


def cross_entropy(output, label, derivative=False, total=False):
    log = np.vectorize(m.log)

    if derivative:
        value = -1 * (label / output) + (1 - label) / (1 - output)
    else:
        value = -1 * (label * log(output) + (1 - label) * log(1 - output))

    if total:
        value = np.sum(value)
    else:
        pass
    return value
