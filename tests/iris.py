import numpy as np
from dsk.data_sets import iris
from dsk.neural_network.models import mlp
from dsk import preprocessing
from dsk.preprocessing import train_test_split
from matplotlib import pyplot as plt


def main():
    df = iris
    X = df.iloc[:, 1:5].values
    y = df.iloc[:, -1].values
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    np.random.seed(0)
    nn = mlp.MLP(cost_function='mse', learning_rate=0.0001)
    nn.add_layer(mlp.InputLayer(4, activation_function='relu'))
    nn.add_layer(mlp.PerceptronLayer(2, activation_function='relu'))
    nn.add_layer(mlp.PerceptronLayer(2, activation_function='relu'))
    nn.add_layer(mlp.OutputLayer(2, activation_function='relu'))
    nn.train(X_train, y_train, epochs=500)

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
