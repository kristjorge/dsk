import math as m


def leaky_relu(activation, derivative=False):
    factor = 0.01
    if not derivative:
        if activation > 0:
            return activation
        else:
            return factor*activation
    else:
        if activation > 0:
            return 1
        else:
            return factor


def relu(activation, derivative=False):
    if not derivative:
        return max(activation, 0.)
    else:
        if activation > 0:
            return 1
        else:
            return 0


def sigmoid(activation, derivative=False):
    transformation = 1. / (1 + m.exp(-activation))
    if not derivative:
        return transformation
    else:
        return transformation * (1 - transformation)


def tanh(activation, derivative=False):
    transformation = 2. / (1 + m.exp(-2*activation)) - 1
    if not derivative:
        return transformation
    else:
        return 1 - transformation**2


def arctan(activation, derivative=False):
    if not derivative:
        return m.atan(activation)
    else:
        return 1 / (1 + activation**2)


def linear(activation, derivative=False):

    if not derivative:
        return activation
    else:
        return 1


