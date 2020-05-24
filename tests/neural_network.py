import numpy as np
from dsk.neural_network.models.sequential.sequential import Sequential
from dsk.neural_network.models.sequential.layers import PerceptronLayer
from matplotlib import pyplot as plt


def main():

    # x_train = [np.array([0, 0, 0, 0]),  # 0.0
    #            np.array([0, 0, 0, 1]),  # 0.1
    #            np.array([0, 0, 1, 0]),  # 0.2
    #            np.array([0, 0, 1, 1]),  # 0.3
    #            np.array([0, 1, 0, 0]),  # 0.4
    #            np.array([0, 1, 0, 1]),  # 0.5
    #            np.array([0, 1, 1, 0]),  # 0.6
    #            np.array([0, 1, 1, 1]),  # 0.7
    #            np.array([1, 0, 0, 0]),  # 0.8
    #            np.array([1, 0, 0, 1]),  # 0.9
    #            np.array([1, 0, 1, 0])   # 0.10
    #            ]
    #
    # y_train = [np.array([0.0]),  # np.array([0, 0, 0, 0]
    #            np.array([0.1]),  # np.array([0, 0, 0, 1]
    #            np.array([0.2]),  # np.array([0, 0, 1, 0]
    #            np.array([0.3]),  # np.array([0, 0, 1, 1]
    #            np.array([0.4]),  # np.array([0, 1, 0, 0]
    #            np.array([0.5]),  # np.array([0, 1, 0, 1]
    #            np.array([0.6]),  # np.array([0, 1, 1, 0]
    #            np.array([0.7]),  # np.array([0, 1, 1, 1]
    #            np.array([0.8]),  # np.array([1, 0, 0, 0]
    #            np.array([0.9]),  # np.array([1, 0, 0, 1]
    #            np.array([0.10]),  # np.array([1, 0, 1, 0]
    #            ]

    x_train = [[2*i/1000] for i in range(500)]
    y_train = [[x[0]] for x in x_train]

    x_pred = [[(20*i + 10)/1000] for i in range(50)]
    y_pred = [[x[0]] for x in x_pred]
    predictions = []

    np.random.seed(0)
    nn = Sequential(input_size=1, output_size=1, cost_function='mse', learning_rate=0.1)
    nn.add_layer(PerceptronLayer(3, activation_function='sigmoid', dropout=0.0))
    nn.add_layer(PerceptronLayer(3, activation_function='sigmoid', dropout=0.0))
    nn.train(x_train, y_train, epochs=10)

    for x in x_pred:
        predictions.append(nn.predict(x))

    plt.scatter(x_train, y_train, c='r')
    plt.scatter(x_pred, y_pred, c='g')
    plt.scatter(x_pred, predictions, c='b')
    plt.show()


if __name__ == "__main__":
    main()
