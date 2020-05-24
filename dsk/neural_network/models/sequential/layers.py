from dsk.neural_network.activation_functions import get_activation_function
import numpy as np


class InputLayer:

    def __init__(self, size, activation_function='linear'):
        self.size = size
        self.network = None
        self.layer_no = None
        self.activation_function = get_activation_function(activation_function)
        self.z = None
        self.h = None
        self.b = None
        self.error = None
        self.bias_gradient = None
        self._errors = []
        self._bias_gradients = []

    def set_input_activations(self, z):
        self.z = z
        self.h = self.activation_function(self.z + self.b)

    def initialise(self, network, layer_no):
        self.network = network
        self.layer_no = layer_no
        self.z = np.random.random((self.size, 1))
        self.b = np.random.random((self.size, 1))
        self.h = self.activation_function(self.z + self.b)

    def reset_gradients(self):
        self._bias_gradients = []
        self._errors = []

    def forward_propagation(self):
        pass

    def backward_propagation(self):
        error = np.dot(self.next_layer.w.T, self.next_layer.error)
        error = np.multiply(error, self.activation_function(self.z, derivative=True))
        self.error = error
        self._errors.append(error)
        self.bias_gradient = error
        self._bias_gradients.append(self.error)

    def adjust_with_gradients(self, learning_rate):
        b = np.zeros(self.b.shape)
        for grad in self._bias_gradients:
            b = np.add(b, grad)
        b = b / len(self._bias_gradients)
        self.b -= learning_rate * b

    @property
    def next_layer(self):
        return self.network.layers[self.layer_no + 1]


class OutputLayer:
    def __init__(self, size, activation_function='relu'):

        self.size = size
        self.network = None
        self.layer_no = None
        self._errors = []
        self._weight_gradients = []
        self._bias_gradients = []
        self.target_output = None
        self.h = None
        self.b = None
        self.z = None
        self.w = None
        self.error = None
        self.weight_gradient = None
        self.bias_gradient = None
        self.activation_function = get_activation_function(activation_function)

    def initialise(self, network, layer_no):
        previous_layer = network.layers[layer_no - 1]
        self.layer_no = layer_no
        self.network = network
        self.z = np.random.random((self.size, 1))
        self.b = np.random.random((self.size, 1))
        self.w = np.random.random((self.size, previous_layer.size))
        self.h = self.activation_function(self.z + self.b)

    def reset_gradients(self):
        self._bias_gradients = []
        self._weight_gradients = []
        self._errors = []

    def forward_propagation(self):
        self.z = np.dot(self.w, self.previous_layer.h)
        self.h = self.activation_function(self.z + self.b)

    def backward_propagation(self):
        error = self.network.cost_function(self.h, self.target_output, derivative=True)
        error = np.multiply(error, self.activation_function(self.z, derivative=True))

        self.error = error
        self._errors.append(error)

        self.weight_gradient = np.dot(self.error, self.previous_layer.h.T)
        self._weight_gradients.append(self.weight_gradient)

        self.bias_gradient = error
        self._bias_gradients.append(self.error)

    def adjust_with_gradients(self, learning_rate):
        b = np.zeros(self.b.shape)
        for grad in self._bias_gradients:
            b = np.add(b, grad)
        b = b / len(self._bias_gradients)
        self.b -= learning_rate * b

        w = np.zeros(self.w.shape)
        for grad in self._weight_gradients:
            w = np.add(w, grad)
        w = w / len(self._weight_gradients)
        self.w -= learning_rate * w

    @property
    def total_error(self):
        return np.sum(self.network.cost_function(self.h, self.target_output, derivative=False))

    @property
    def previous_layer(self):
        return self.network.layers[self.layer_no - 1]


class PerceptronLayer:

    def __init__(self, size, activation_function='linear', dropout=0.):

        self.activation_function = get_activation_function(activation_function)
        self.size = size
        self.network = None
        self.layer_no = None
        self._dropout = dropout
        self._errors = []
        self._weight_gradients = []
        self._bias_gradients = []
        self.z = None
        self.h = None
        self.b = None
        self.w = None
        self.error = None
        self.weight_gradient = None
        self.bias_gradient = None

    def initialise(self, network, layer_no):
        # Initialising with random weights and biases
        self.network = network
        self.layer_no = layer_no
        self.z = np.random.random((self.size, 1))
        self.b = np.random.random((self.size, 1))
        self.w = np.random.random((self.size, self.previous_layer.size))
        self.h = self.activation_function(self.z + self.b)

    def reset_gradients(self):
        self._bias_gradients = []
        self._weight_gradients = []
        self._errors = []

    def forward_propagation(self):
        previous_layer = self.network.layers[self.layer_no - 1]
        self.z = np.dot(self.w, previous_layer.h)
        self.h = self.activation_function(self.z + self.b)

    def backward_propagation(self):
        error = np.dot(self.next_layer.w.T, self.next_layer.error)
        diff_act_func = self.activation_function(self.z, derivative=True)
        error = np.multiply(error, diff_act_func)
        self.error = error
        self._errors.append(error)
        self.weight_gradient = np.dot(self.error, self.previous_layer.h.T)
        self.bias_gradient = error
        self._weight_gradients.append(self.weight_gradient)
        self._bias_gradients.append(self.error)

    def adjust_with_gradients(self, learning_rate):
        b = np.zeros(self.b.shape)
        for grad in self._bias_gradients:
            b = np.add(b, grad)
        b = b / len(self._bias_gradients)
        self.b -= learning_rate * b

        w = np.zeros(self.w.shape)
        for grad in self._weight_gradients:
            w = np.add(w, grad)
        w = w / len(self._weight_gradients)
        self.w -= learning_rate * w

    @property
    def previous_layer(self):
        return self.network.layers[self.layer_no - 1]

    @property
    def next_layer(self):
        return self.network.layers[self.layer_no + 1]
