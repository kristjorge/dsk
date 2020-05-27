from dsk.neural_network.activation_functions import get_activation_function
import numpy as np


class PerceptronLayer:

    def __init__(self, size, activation_function='relu'):
        self.activation_function = get_activation_function(activation_function)
        self.size = size
        self.network = None
        self.layer_no = None
        self._weight_gradients = []
        self._bias_gradients = []
        self.z = None
        self.h = None
        self.b = None
        self.w = None
        self.error = None

    def initialise(self, network, layer_no, mean=0, stdev=1):
        self.network = network
        self.layer_no = layer_no
        self.z = np.random.normal(mean, stdev, (self.size, 1))
        self.b = np.random.normal(mean, stdev, (self.size, 1))
        self.w = np.random.normal(mean, stdev, (self.size, self.previous_layer.size))
        self.h = self.activation_function(self.z + self.b)

    def reset_gradients(self):
        self._bias_gradients = []
        self._weight_gradients = []

    def forward_propagation(self):
        self.z = np.dot(self.w, self.previous_layer.h)
        self.h = self.activation_function(self.z + self.b)

    def backward_propagation(self):
        self.error = np.dot(self.next_layer.w.T, self.next_layer.error)
        self.error = np.multiply(self.error, self.activation_function(self.z, derivative=True))
        self._weight_gradients.append(np.dot(self.error, self.previous_layer.h.T))
        self._bias_gradients.append(self.error)

    def adjust_with_gradients(self):
        if hasattr(self, 'b'):
            b = np.zeros(self.b.shape)
            for grad in self._bias_gradients:
                b = np.add(b, grad)
            b = b / len(self._bias_gradients)
            self.b -= self.network.learning_rate * b

        if hasattr(self, 'w'):
            w = np.zeros(self.w.shape)
            for grad in self._weight_gradients:
                w = np.add(w, grad)
            w = w / len(self._weight_gradients)
            self.w -= self.network.learning_rate * w

    @property
    def previous_layer(self):
        try:
            return self.network.layers[self.layer_no - 1]
        except IndexError:
            pass

    @property
    def next_layer(self):
        try:
            return self.network.layers[self.layer_no + 1]
        except IndexError:
            pass


class InputLayer(PerceptronLayer):

    def __init__(self, size, activation_function='linear'):
        super().__init__(size, activation_function)
        del self.w

    def set_input_activations(self, z):
        self.z = z
        self.h = self.activation_function(self.z + self.b)

    def initialise(self, network, layer_no, mean=0, stdev=1):
        self.network = network
        self.layer_no = layer_no
        self.z = np.random.normal(0, 1, (self.size, 1))
        self.b = np.random.normal(0, 1, (self.size, 1))
        self.h = self.activation_function(self.z + self.b)

    def reset_gradients(self):
        self._bias_gradients = []

    def forward_propagation(self):
        pass

    def backward_propagation(self):
        self.error = np.dot(self.next_layer.w.T, self.next_layer.error)
        self.error = np.multiply(self.error, self.activation_function(self.z, derivative=True))
        self._bias_gradients.append(self.error)

    @property
    def next_layer(self):
        return self.network.layers[self.layer_no + 1]


class OutputLayer(PerceptronLayer):
    def __init__(self, size, activation_function='relu'):
        super().__init__(size, activation_function)
        self.target_output = None

    def backward_propagation(self):
        self.error = self.network.cost_function(self.h, self.target_output, derivative=True)
        self.error = np.multiply(self.error, self.activation_function(self.z, derivative=True))
        self._weight_gradients.append(np.dot(self.error, self.previous_layer.h.T))
        self._bias_gradients.append(self.error)

    @property
    def total_error(self):
        return np.sum(self.network.cost_function(self.h, self.target_output, derivative=False))

    @property
    def previous_layer(self):
        return self.network.layers[self.layer_no - 1]


