import numpy as np
from dsk.neural_network.models import mlp
from dsk.neural_network.layers import layers
from dsk.preprocessing import train_test_split
from dsk.metrics.costs.mean_squared_error import mse
from matplotlib import pyplot as plt


def main():

    X = np.array([[np.random.random()] for _ in range(200)])
    y = 2*X

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # np.random.seed(0)
    nn = mlp.Sequential(cost_function='mse', learning_rate=0.1)
    nn.add_layer(layers.InputLayer(1, activation_function='relu'))
    nn.add_layer(layers.PerceptronLayer(8, activation_function='relu'))
    nn.add_layer(layers.PerceptronLayer(5, activation_function='relu'))
    nn.add_layer(layers.OutputLayer(1, activation_function='relu'))
    nn.train(X_train, y_train, epochs=200)

    f_predicted = []
    for x in X:
        f_predicted.append(nn.predict(x)[0])
    f = np.array(f_predicted)

    plt.scatter(X, y)
    plt.scatter(X, f)
    plt.show()

    plt.plot(nn.costs)
    plt.show()


if __name__ == '__main__':
    main()
