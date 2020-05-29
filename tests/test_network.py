import numpy as np
from dsk.neural_network.models import mlp
from dsk.neural_network.layers import layers
from matplotlib import pyplot as plt


def main():

    X = np.array([[1], [1]])
    y = np.array([[1], [1]])

    # X_train, X_test, y_train, y_test = train_test_split(X, y)

    # np.random.seed(0)
    nn = mlp.MLP(cost_function='mse', learning_rate=0.5)
    nn.add_layer(layers.InputLayer(2, activation_function='relu'))
    nn.add_layer(layers.PerceptronLayer(2, activation_function='sigmoid'))
    nn.add_layer(layers.PerceptronLayer(2, activation_function='sigmoid'))
    nn.add_layer(layers.OutputLayer(2, activation_function='relu'))
    nn.train(X, y, epochs=400)

    plt.plot(nn.average_costs)
    plt.show()
    print(nn.predict(X))

if __name__ == '__main__':
    main()
