import math as m
import numpy as np


def get_activation_function(activation_function):
    if activation_function == 'linear':
        return np.vectorize(linear)

    elif activation_function == 'sigmoid':
        return np.vectorize(sigmoid)

    elif activation_function == 'relu':
        return np.vectorize(relu)

    elif activation_function == 'tanh':
        return np.vectorize(tanh)


def relu(activation, derivative=False):

    """
    activation:         The input activation to the neuron

    method transform:   Returns the output from the activation function given the input activation
    method prime:       Returns the derivative of the activation function wrt to the input activation.
                        Used in the backward passing
    """

    if not derivative:
        return max(0, activation)
    else:
        if activation > 0:
            return 1
        else:
            return 0


def sigmoid(activation, derivative=False):

    """
    activation:         The input activation to the neuron

    method transform:   Returns the output from the activation function given the input activation
    method prime:       Returns the derivative of the activation function wrt to the input activation.
                        Used in the backward passing
    """
    transformation = 1 / (1 + m.exp(-activation))
    if not derivative:
        return transformation

    else:
        return transformation * (1 - transformation)


def tanh(activation, derivative=False):

    """
    activation:         The input activation to the neuron

    method transform:   Returns the output from the activation function given the input activation
    method prime:       Returns the derivative of the activation function wrt to the input activation.
                        Used in the backward passing
    """

    # transformation = (m.exp(activation) - m.exp(-activation)) / (m.exp(activation) + m.exp(-activation))
    transformation = 2 / (1 + m.exp(-2*activation)) - 1
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

    """
    activation:         The input activation to the neuron

    method transform:   Returns the output from the activation function given the input activation
    method prime:       Returns the derivative of the activation function wrt to the input activation.
                        Used in the backward passing
    """

    if not derivative:
        return activation

    else:
        return 1

