import numpy as np
from matplotlib import pyplot as plt
from dsk.data_sets import iris
from dsk.neural_network.models import feed_forward
from dsk.neural_network.layers import layers
from dsk.preprocessing import encoding, feature_scaling, model_selection
from dsk.neural_network.initialization.initializer import XavierInitializer
from dsk.neural_network.activation import softmax, sigmoid, linear, tanh
from dsk.metrics.costs import mse, cross_entropy
from dsk.preprocessing.feature_scaling import Normalizer
from dsk.preprocessing.model_selection import shuffle


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

    nn = feed_forward.FeedForward(cost_function=cross_entropy, learning_rate=0.05, initialisation=XavierInitializer)
    nn.add_layer(layers.InputLayer(4, activation_function=linear))
    nn.add_layer(layers.FullyConnectedLayer(20, activation_function=tanh))
    nn.add_layer(layers.FullyConnectedLayer(20, activation_function=tanh))
    nn.add_layer(layers.FullyConnectedLayer(20, activation_function=tanh))
    nn.add_layer(layers.FullyConnectedLayer(20, activation_function=tanh))
    nn.add_layer(layers.FullyConnectedLayer(20, activation_function=tanh))
    nn.add_layer(layers.OutputLayer(3, activation_function=softmax))
    nn.train(X_train, y_train, epochs=1)

    # plt.plot(nn.average_costs)
    # plt.show()

    predictions = []
    predictions_labeled = []
    for x_row, y_row in zip(X_test, y_test):
        p = nn.predict(x_row).T
        prediction = np.zeros((p.shape[0], p.shape[1]), dtype=int)
        prediction[0, np.argmax(p)] = 1
        predictions.append(prediction)
        label = one_hot.inverse_transform(predictions[-1])
        predictions_labeled.append(le.inverse_transform(label))

    for label, pred, x, y in zip(predictions_labeled, predictions, X_test, y_test):
        test_label = one_hot.inverse_transform(y)
        test_label = le.inverse_transform(test_label)
        print('Input activations: {}'.format(x))
        print('Prediction: {}'.format(pred))
        print('Predicted label: {}'.format(label))
        print('Test label: {}'.format(test_label))
        print('------- ------- ------- ------')


if __name__ == '__main__':
    main()
