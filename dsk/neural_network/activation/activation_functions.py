import math as m
import numpy as np


def leaky_relu(input_activation, derivative=False):

    def transformation(a):
        nonlocal derivative
        factor = 0.01
        if not derivative:
            if a > 0:
                return a
            else:
                return a * factor
        else:
            if a > 0:
                return 1
            else:
                return factor

    output_activation = np.zeros((input_activation.shape[0], input_activation.shape[1]))
    for row in range(input_activation.shape[0]):
        for col in range(input_activation.shape[1]):
            output_activation[row, col] = transformation(input_activation[row, col])

    return output_activation


def relu(input_activation, derivative=False):

    def transformation(a):
        nonlocal derivative
        if not derivative:
            return max(a, 0.)
        else:
            if a > 0:
                return 1
            else:
                return 0

    output_activation = np.zeros((input_activation.shape[0], input_activation.shape[1]))
    for row in range(input_activation.shape[0]):
        for col in range(input_activation.shape[1]):
            output_activation[row, col] = transformation(input_activation[row, col])

    return output_activation


def sigmoid(input_activation, derivative=False):

    def transformation(a):
        nonlocal derivative
        sig = 1. / (1 + m.exp(-a))
        if not derivative:
            return sig
        else:
            return sig * (1 - sig)

    output_activation = np.zeros((input_activation.shape[0], input_activation.shape[1]))
    for row in range(input_activation.shape[0]):
        for col in range(input_activation.shape[1]):
            output_activation[row, col] = transformation(input_activation[row, col])

    return output_activation


def tanh(input_activation, derivative=False):
    def transformation(a):
        nonlocal derivative
        sig = 2. / (1 + m.exp(-2*a)) - 1
        if not derivative:
            return sig
        else:
            return 1 - sig**2

    output_activation = np.zeros((input_activation.shape[0], input_activation.shape[1]))
    for row in range(input_activation.shape[0]):
        for col in range(input_activation.shape[1]):
            output_activation[row, col] = transformation(input_activation[row, col])

    return output_activation


def arctan(input_activation, derivative=False):
    def transformation(a):
        nonlocal derivative
        if not derivative:
            return m.atan(a)
        else:
            return 1 / (1 + a**2)

    output_activation = np.zeros((input_activation.shape[0], input_activation.shape[1]))
    for row in range(input_activation.shape[0]):
        for col in range(input_activation.shape[1]):
            output_activation[row, col] = transformation(input_activation[row, col])

    return output_activation


def softmax(input_activation, derivative=False):

    def transformation(a):
        nonlocal derivative, input_activation
        logit_transformation = np.exp(a) / np.sum(np.exp(input_activation))
        if not derivative:
            return logit_transformation
        else:
            return logit_transformation * (1 - logit_transformation)

    output_activation = np.zeros((input_activation.shape[0], input_activation.shape[1]))
    for row in range(input_activation.shape[0]):
        for col in range(input_activation.shape[1]):
            output_activation[row, col] = transformation(input_activation[row, col])

    return output_activation


def linear(input_activation, derivative=False):

    if not derivative:
        return input_activation
    else:
        return np.ones((input_activation.shape[0], input_activation.shape[1]))



