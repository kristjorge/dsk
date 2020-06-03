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
        self._weights = []
        self._biases = []
        self.z = None
        self.b = None
        self.w = None
        self.error = None

    def initialise(self, network, layer_no, initializer):
        self.network = network
        self.layer_no = layer_no
        self.z = np.ones((self.size, 1))
        if hasattr(self, 'b'):
            self.b = initializer(self).init_b()

        if hasattr(self, 'w'):
            self.w = initializer(self).init_w()

    def reset_gradients(self):
        if hasattr(self, 'b'):
            self._bias_gradients = []
        if hasattr(self, 'w'):
            self._weight_gradients = []

    def forward_propagation(self):
        if self.layer_no > 0:
            self.z = np.dot(self.w, self.previous_layer.h)
        else:
            pass

    def backward_propagation(self):
        self.error = np.dot(self.next_layer.w.T, self.next_layer.error)
        self.error = np.multiply(self.error, self.activation_function(self.z, derivative=True))
        self._weight_gradients.append(np.dot(self.error, self.previous_layer.h.T))
        self._bias_gradients.append(self.error)

    def adjust_with_gradients(self):
        if hasattr(self, 'b'):
            db = np.zeros(self.b.shape)
            for grad in self._bias_gradients:
                db = np.add(db, grad)
            db = db / len(self._bias_gradients)
            self._biases.append(self.b.copy())
            self.b -= self.network.learning_rate * db

        if hasattr(self, 'w'):
            dw = np.zeros(self.w.shape)
            for grad in self._weight_gradients:
                dw = np.add(dw, grad)
            dw = dw / len(self._weight_gradients)
            self._weights.append(self.w.copy())
            self.w -= self.network.learning_rate * dw

    @property
    def h(self):
        return self.activation_function(self.z + self.b)

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
        del self.b

    def set_input_activations(self, z):
        self.z = z

    def backward_propagation(self):
        pass

    @property
    def h(self):
        return self.activation_function(self.z)


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
        return self.network.cost_function(self.h, self.target_output, derivative=False, total=True)


