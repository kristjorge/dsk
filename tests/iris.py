import numpy as np
from dsk.data_sets import iris
from dsk.neural_network.models import sequential
from dsk import preprocessing
from dsk.preprocessing import train_test_split
from matplotlib import pyplot as plt
import sklearn.model_selection
import sklearn.preprocessing


def main():
    df = iris
    X = df.iloc[:, 1:4].values
    y = df.iloc[:, -1].values
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    nn = sequential.Sequential(cost_function='mse', learning_rate=0.1)
    nn.add_layer(sequential.InputLayer(3, activation_function='relu'))
    nn.add_layer(sequential.OutputLayer(1, activation_function='relu'))
    nn.add_layer(sequential.PerceptronLayer(3, activation_function='relu'))
    nn.train(X_train, y_train, epochs=50)

    plt.plot(nn.costs)
    plt.show()

    for x_row, y_row in zip(X_test, y_test):
        prediction = nn.predict(x_row)
        print('Input activations: {}'.format(x_row))
        print('Prediction: {}'.format(prediction))
        print('Test label: {}'.format(y_row))
        print('------- ------- ------- ------')


if __name__ == '__main__':
    main()
