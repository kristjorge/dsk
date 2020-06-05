import dsk.metrics
import dsk.metrics.costs
from dsk.preprocessing.model_selection import shuffle
import numpy as np
import math as m


class LinearRegression:

    def __init__(self, epochs=50, learning_rate=0.1):
        self._epochs = epochs
        self._lr = learning_rate
        self._mini_batch_size = None
        self.coefficients = []
        self.loss = []
        self.loss_function = dsk.metrics.costs.mse
        self.features = []
        self.R = None

    def fit(self, X, y, mini_batch_size=None):
        """
        Fits coefficients in a multivariate linear expression to fit the training data. The number of variables is
        inferred from the dimensions of the X matrix provided in the training set
        :param X:
        :param y:
        :return:
        """

        self._mini_batch_size = mini_batch_size

        # Transforming X to an m x n matrix
        if X.ndim == 1:
            self.features.append(X.reshape(-1, 1))
        else:
            for col in range(X.shape[1]):
                self.features.append(X[:, col].reshape(-1, 1))

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Coefficients
        for feature_no in range(len(self.features)):
            self.coefficients.append(RegressionCoefficient())

        # Intercept
        self.coefficients.append(RegressionCoefficient())

        if self._mini_batch_size:
            features_batches, label_batches = self._sample_batches(X, y)
        else:
             features_batches, label_batches = [X], [y]

        for _ in range(self._epochs):

            # If mini batch is provided, get a subset of the total features matrix. If not, use self.features
            for batch_no, (features, labels) in enumerate(zip(features_batches, label_batches)):
                # Calculate function value
                f = self._calc_expression(features)

                # Store the loss value
                self.loss.append(self.loss_function(f, labels, total=True))

                # Updating coefficients
                self._update_with_gradients(f, features, labels)

        f_fitted = self.predict(X)
        self.R = dsk.metrics.r_squared(y, f_fitted)

    def predict(self, X):

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[1] + 1 != len(self.coefficients):
            quit('Dimensions do not align')

        # Calculate function value
        f = self._calc_expression(X)

        return f

    def _calc_expression(self, features):
        f = self.coefficients[-1].value
        for idx, c in enumerate(self.coefficients):
            if idx < len(self.coefficients) - 1:
                f += c.value * features[:, idx].reshape(-1, 1)
        return f

    def _update_with_gradients(self, f, X, y):
        for idx, c in enumerate(self.coefficients):
            self.coefficients[idx].log.append(self.coefficients[idx].value)
            if idx == len(self.coefficients)-1:
                gradient = self._lr * np.mean(self.loss_function(f, y, derivative=True))
            else:
                gradient = self._lr * np.mean(np.multiply(X[:, idx].reshape(-1, 1), self.loss_function(f, y, derivative=True)))
            self.coefficients[idx].value -= gradient
            self.coefficients[idx].gradients.append(gradient)

    def _sample_batches(self, X, y):
        X_shuffled = X.copy()
        y_shuffled = y.copy()
        X_shuffled, y_shuffled = shuffle(X_shuffled, y_shuffled)

        num_batches = m.floor(X.shape[0] / self._mini_batch_size)
        X_batches = [np.array(X_shuffled[i:i+self._mini_batch_size, :]) for i in range(num_batches)]
        y_batches = [np.array(y_shuffled[i:i+self._mini_batch_size, :]) for i in range(num_batches)]

        return X_batches, y_batches

class RegressionCoefficient:

    def __init__(self):
        self.value = np.random.random()
        self.log = []
        self.gradients = []
