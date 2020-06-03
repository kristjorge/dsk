import dsk.metrics
import dsk.metrics.costs
import numpy as np


class LinearRegression:

    def __init__(self, epochs=50, learning_rate=0.1):
        self._epochs = epochs
        self._lr = learning_rate
        self.coefficients = []
        self.loss = []
        self.loss_function = dsk.metrics.costs.mse
        self.features = []
        self.R = None

    def fit(self, X, y):
        """
        Fits coefficients in a multivariate linear expression to fit the training data. The number of variables is
        inferred from the dimensions of the X matrix provided in the training set
        :param X:
        :param y:
        :return:
        """

        # Transforming X to an m x n matrix
        if X.ndim == 1:
            self.features.append(X.reshape(-1, 1))
        else:
            for col in range(X.shape[1]):
                self.features.append(X[:, col].reshape(-1, 1))

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for feature_no in range(len(self.features)):
            self.coefficients.append(RegressionCoefficient())

        # Intercept
        self.coefficients.append(RegressionCoefficient())

        for _ in range(self._epochs):
            # Calculate function value

            f = self._calc_expression(self.features)
            self.loss.append(self.loss_function(f, y, total=True))

            # Updating coefficients
            for idx, c in enumerate(self.coefficients):
                self.coefficients[idx].log.append(self.coefficients[idx].value)
                if idx == len(self.coefficients)-1:
                    gradient = self._lr * np.mean(self.loss_function(f, y, derivative=True))
                else:
                    gradient = self._lr * np.mean(np.multiply(self.features[idx], self.loss_function(f, y, derivative=True)))
                self.coefficients[idx].value -= gradient
                self.coefficients[idx].gradients.append(gradient)

        f_fitted = self.predict(X)
        self.R = dsk.metrics.r_squared(y, f_fitted)

    def predict(self, X):

        if X.ndim == 1:
            features = [(X.reshape(-1, 1))]
            if len(features) + 1 != len(self.coefficients):
                quit('Dimensions do not align')
        else:
            features = [X[:, col].reshape(-1, 1) for col in range(X.shape[1])]

        if len(features) + 1 != len(self.coefficients):
            quit('Dimensions do not align')

        # Calculate function value
        f = self._calc_expression(features)

        return f

    def _calc_expression(self, features):
        f = self.coefficients[-1].value
        for idx, c in enumerate(self.coefficients):
            if idx < len(self.coefficients) - 1:
                f += c.value * features[idx]
        return f


class RegressionCoefficient:

    def __init__(self):
        self.value = np.random.random()
        self.log = []
        self.gradients = []
