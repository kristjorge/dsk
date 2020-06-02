import numpy as np
from dsk.data_sets import iris
from dsk.neural_network.models import mlp
from dsk.neural_network.layers import layers
from dsk import preprocessing
from dsk.preprocessing import train_test_split
from dsk.neural_network.initialization.initializer import XavierInitializer
from dsk.metrics.costs import mse, cross_entropy
from matplotlib import pyplot as plt


def main():
    X = iris.iloc[:, 1:5].values
    y = iris.iloc[:, -1].values
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    nn = mlp.MLP(cost_function=cross_entropy, learning_rate=0.15, initialisation=XavierInitializer)
    nn.add_layer(layers.InputLayer(4, activation_function='linear'))
    nn.add_layer(layers.PerceptronLayer(10, activation_function='tanh'))
    nn.add_layer(layers.PerceptronLayer(10, activation_function='tanh'))
    nn.add_layer(layers.PerceptronLayer(10, activation_function='tanh'))
    nn.add_layer(layers.PerceptronLayer(10, activation_function='tanh'))
    nn.add_layer(layers.PerceptronLayer(10, activation_function='tanh'))
    nn.add_layer(layers.OutputLayer(4, activation_function='sigmoid'))
    nn.train(X_train, y_train, epochs=200)

    # plt.plot(nn.costs)
    # plt.show()

    for x_row, y_row in zip(X_test, y_test):
        prediction = nn.predict(x_row)
        print('Input activations: {}'.format(x_row))
        print('Prediction: {}'.format(prediction))
        print('Test label: {}'.format(y_row))
        print('------- ------- ------- ------')


if __name__ == '__main__':
    main()
