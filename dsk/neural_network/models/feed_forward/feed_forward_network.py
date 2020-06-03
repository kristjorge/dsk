from dsk.metrics.costs import mse
import dsk.neural_network.layers.layers as mlp_layers
from dsk.neural_network.initialization.initializer import RandomInitializer
from dsk.utils.progress_bar import ProgressBar
import numpy as np


class FeedForward:

    """
    Attributes:
        x_train:            List of training examples
        y_train:            List of training labels
        learning_rate:      The specified learning rate which is multiplied with the gradients to update the weights
                            and biases
        layers:             The different layers in the network

    """

    def __init__(self, learning_rate=0.01, cost_function=mse, initialisation=RandomInitializer):

        """
        :param learning_rate: Learning rate used in gradient descent to update weights and biases
        :param cost_function: Which cost function should be used in computing costs. Currently only MSE is
        implemented
        """

        self.X_train = None
        self.y_train = None
        self.average_costs = []
        self.learning_rate = learning_rate
        self.cost_function = None
        self.layers = []
        self.cost_function = cost_function
        self._initializer = initialisation

    def __len__(self):
        return len(self.layers)

    @property
    def input_layer(self):
        return self.layers[0]

    @property
    def output_layer(self):
        return self.layers[-1]

    def add_layer(self, layer):

        """
        Adds a new layer to the feed_forward model.

        If the layer is of type InputLayer, place it at the start
        If the layer is of type OutputLayer, place it at the end
        If neither, check for whether or not an OutputLayer is at the end.
            If it is: insert it at the next last place.
            If not: append it at the end
        :param layer:
        :return:
        """

        if type(layer) == mlp_layers.InputLayer:
            self.layers.insert(0, layer)
        elif type(layer) == mlp_layers.OutputLayer:
            self.layers.insert(len(self.layers), layer)
        else:
            if type(self.layers[-1]) == mlp_layers.OutputLayer:  # If there is an Output layer
                self.layers.insert(-1, layer)
            else:
                self.layers.append(layer)

    def initialise(self, X_train, y_train):

        """
        :param X_train: List of training examples
        :param y_train: List of training labels
        :param cost_function: Input cost function used in training
        :return: Nothing

        Also runs the initialise method on the layers which runs the initialise method on the Neurons in the layers
        """

        # Forcing x_train and y_train to be 2D matrices
        if X_train.ndim == 1:
            self.X_train = X_train.reshape(-1, 1)
        else:
            self.X_train = X_train

        if y_train.ndim == 1:
            self.y_train = y_train.reshape(-1, 1)
        else:
            self.y_train = y_train

        if len(self.X_train) != len(self.y_train):
            quit("Training features and labels should be the same length\nQuitting...")

        for layer_no, layer in enumerate(self.layers):
            layer.initialise(self, layer_no, self._initializer)

    def train(self, x_train, y_train, epochs):

        # Check if first layer is an InputLayer and last layer is an OutputLayer
        if not type(self.layers[0]) == mlp_layers.InputLayer or not type(self.layers[-1]) == mlp_layers.OutputLayer:
            quit('Model needs to have an input layer and output layer to work\nQuitting...')

        self.initialise(x_train, y_train)
        progress_bar_epochs = ProgressBar(epochs)
        for _ in range(epochs):
            self.reset_gradients()
            costs = []
            for i in range(self.X_train.shape[0]):
                self.forward_pass(i)
                costs.append(self.output_layer.total_error)
                self.backward_pass()
            self.average_costs.append(sum(costs) / len(costs))
            self.adjust_weights_and_biases()
            progress_bar_epochs.update()

    def predict(self, a):
        if type(a) == np.ndarray and a.ndim == 1:
            a = a.reshape(-1, 1)

        self.input_layer.set_input_activations(a)
        for layer in self.layers:
            layer.forward_propagation()

        return self.output_layer.h

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()

    def forward_pass(self, training_sample_no):

        training_input = self.X_train[training_sample_no, :]
        training_output = self.y_train[training_sample_no, :]
        if type(training_input) == list:
            training_input = np.array(training_input).reshape(-1, 1)
        elif type(training_input) == np.ndarray and len(training_input.shape) == 1:
            training_input = training_input.reshape(-1, 1)

        if type(training_output) == list:
            training_output = np.array(training_output).reshape(-1, 1)
        elif type(training_input) == np.ndarray and len(training_output.shape) == 1:
            training_output = training_output.reshape(-1, 1)

        self.input_layer.set_input_activations(training_input)
        self.output_layer.target_output = training_output
        for layer in self.layers:
            layer.forward_propagation()

    def backward_pass(self):
        for i in range(len(self.layers)-1, -1, -1):
            self.layers[i].backward_propagation()

    def adjust_weights_and_biases(self):
        for layer in self.layers:
            layer.adjust_with_gradients()
