from dsk.neural_network.models import feed_forward
from dsk.neural_network.layers import layers
from dsk.neural_network import activation
from dsk.metrics.costs import mse
from dsk.neural_network.initialization.initializer import XavierInitializer
from dsk.preprocessing.model_selection import train_test_split
from matplotlib import pyplot as plt
from dsk.data_sets import simple_set_200 as data_set


def main():

    X = data_set.iloc[:, 0].values
    y = data_set.iloc[:, 3].values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # np.random.seed(0)
    nn = feed_forward.FeedForward(cost_function=mse, learning_rate=0.15, initialisation=XavierInitializer)
    nn.add_layer(layers.InputLayer(1, activation_function=activation.linear))
    nn.add_layer(layers.FullyConnectedLayer(10, activation_function=activation.tanh))
    nn.add_layer(layers.FullyConnectedLayer(10, activation_function=activation.tanh))
    nn.add_layer(layers.FullyConnectedLayer(10, activation_function=activation.tanh))
    nn.add_layer(layers.FullyConnectedLayer(10, activation_function=activation.tanh))
    nn.add_layer(layers.FullyConnectedLayer(10, activation_function=activation.tanh))
    nn.add_layer(layers.OutputLayer(1, activation_function=activation.linear))
    nn.train(X_train, y_train, epochs=200)

    predictions = []
    for x in X_test:
        predictions.append(nn.predict(x)[0])

    plt.figure()
    plt.subplot(211)
    plt.plot(nn.average_costs)
    plt.subplot(212)
    plt.scatter(X, y, s=1)
    plt.scatter(X_test, predictions, s=5)
    plt.show()


if __name__ == '__main__':
    main()
