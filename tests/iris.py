import numpy as np
from dsk.data_sets import iris
from dsk.neural_network.models import mlp
from dsk.neural_network.layers import layers
from dsk.preprocessing import encoding, feature_scaling, model_selection
from dsk.neural_network.initialization.initializer import XavierInitializer
from dsk.metrics.costs import mse, cross_entropy
from dsk.preprocessing.feature_scaling import Normalizer


def main():

    one_hot = encoding.OneHotEncoder()
    le = encoding.LabelEncoder()
    norm = Normalizer()

    X = iris.iloc[:, 1:5].values
    y = iris.iloc[:, -1].values
    le.fit(y)
    y = le.transform(y)
    one_hot.fit(y, [0])
    y = one_hot.transform(y)
    norm.fit(X)
    X = norm.transform(X)


    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    nn = mlp.MLP(cost_function=cross_entropy, learning_rate=0.15, initialisation=XavierInitializer)
    nn.add_layer(layers.InputLayer(4, activation_function='linear'))
    nn.add_layer(layers.PerceptronLayer(10, activation_function='relu'))
    nn.add_layer(layers.PerceptronLayer(10, activation_function='relu'))
    nn.add_layer(layers.PerceptronLayer(10, activation_function='relu'))
    nn.add_layer(layers.PerceptronLayer(10, activation_function='relu'))
    nn.add_layer(layers.PerceptronLayer(10, activation_function='relu'))
    nn.add_layer(layers.OutputLayer(3, activation_function='sigmoid'))
    nn.train(X_train, y_train, epochs=1)

    predictions = []
    predictions_labeled = []
    for x_row, y_row in zip(X_test, y_test):
        predictions.append(nn.predict(x_row).T)
        label = one_hot.inverse_transform(predictions[-1])
        predictions_labeled.append(le.inverse_transform(label))

    for label, pred, x, y in zip(predictions_labeled, predictions, X_test, y_test):
        print('Input activations: {}'.format(x))
        print('Prediction: {}'.format(pred))
        print('Prediction label: {}'.format(label))
        print('Test label: {}'.format(y))
        print('------- ------- ------- ------')


if __name__ == '__main__':
    main()
