from dsk.metrics.costs import mse
from dsk.neural_network.models import sequential
from dsk.utils.progress_bar import ProgressBar
import numpy as np


class Sequential:

    """
    Attributes:
        x_train:            List of training examples
        y_train:            List of training labels
        cost:               Calculated cost for one training example. Returned as a list where cost[0] is the total
                            cost, and cost[1] is the error array
        del_cost:           Derivative of the cost function at the output layer
        learning_rate:      The specified learning rate which is multiplied with the gradients to update the weights
                            and biases
        layers:             The different layers in the network

    """

    def __init__(self, learning_rate=0.01, cost_function='mse'):

        """

        :param learning_rate: Learning rate used in gradient descent to update weights and biases
        :param cost_function: Which cost function should be used in computing costs. Currently only MSE is
        implemented
        """

        super().__init__()
        self.x_train = None
        self.y_train = None
        self.costs = []
        self.learning_rate = learning_rate
        self.cost_function = None
        self.layers = []
        self.progress_bar = None
        if cost_function == 'mse':
            self.cost_function = mse
        else:
            print("No other cost function yet implemented..\nSetting it to 'mse'")
            self.cost_function = mse

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
        Adds a new layer to the sequential model.

        If the layer is of type InputLayer, place it at the start
        If the layer is of type OutputLayer, place it at the end
        If neither, check for whether or not an OutputLayer is at the end.
            If it is: insert it at the next last place.
            If not: append it at the end
        :param layer:
        :return:
        """

        if type(layer) == sequential.InputLayer:
            self.layers.insert(0, layer)
        elif type(layer) == sequential.OutputLayer:
            self.layers.insert(-1, layer)
        else:
            if type(self.layers[-1]) == sequential.OutputLayer:  # If there is an Output layer
                self.layers.insert(-1, layer)
            else:
                self.layers.append(layer)

    def initialise(self, x_train, y_train):

        """
        :param x_train: List of training examples
        :param y_train: List of training labels
        :param cost_function: Input cost function used in training
        :return: Nothing

        Also runs the initialise method on the layers which runs the initialise method on the Neurons in the layers
        """

        self.x_train = x_train
        self.y_train = y_train
        assert len(self.x_train) == len(self.y_train), \
            "Training features and labels should be the same length\nQuitting..."

        for layer_no, layer in enumerate(self.layers):
            layer.initialise(self, layer_no)

    def compute_avg_costs(self, costs):
        avg_cost = sum(costs) / len(costs)
        self.costs.append(avg_cost)

    def train(self, x_train, y_train, epochs):

        # Check if first layer is an InputLayer and last layer is an OutputLayer
        if not type(self.layers[0]) == sequential.InputLayer or not type(self.layers[-1]) == sequential.OutputLayer:
            quit('Model needs to have an input layer and output layer to work\nQuitting...')

        self.initialise(x_train, y_train)
        self.progress_bar = ProgressBar(epochs)
        for _ in range(epochs):
            self.reset_gradients()
            costs = []
            for i in range(self.x_train.shape[0]):
                self.forward_pass(i)
                costs.append(self.output_layer.total_error)
                self.backward_pass()
            self.compute_avg_costs(costs)
            self.adjust_weights_and_biases()
            self.progress_bar.update()

    def predict(self, a):
        # If list, make it an np.array and reshape to a column vector
        if type(a) == list:
            a = np.array(a).reshape(-1, 1)

        # If an np.ndarray with no shape = (x,), the reshape to column vector
        elif type(a) == np.ndarray and len(a.shape) == 1:
            a = a.reshape(-1, 1)

        self.input_layer.set_input_activations(a)
        for layer in self.layers:
            layer.forward_propagation()

        return self.output_layer.h

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()

    def forward_pass(self, training_sample_no):

        training_input = self.x_train[training_sample_no, :]
        training_output = self.y_train[training_sample_no]
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
            layer.adjust_with_gradients(self.learning_rate)
