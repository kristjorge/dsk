import numpy as np
import dsk.costs


class LinearRegression:

    def __init__(self, epochs=50, learning_rate=0.1):
        self._epochs = epochs
        self._learning_rate = learning_rate
        self.coefficients = {}
        self.mse = []
        self.cost_function = dsk.costs.mse
        self.features = []

    def fit(self, X, y):

        # Transforming X to an m x n matrix
        if len(X.shape) == 1:  # Only (M,) size
            self.features.append(X.reshape(-1, 1))
        else:
            for col in range(X.shape[0]):
                self.features.append(X[col].reshape(-1, 1))

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        self.coefficients['intercept'] = np.random.random()
        for feature_no in range(len(self.features)):
            self.coefficients[feature_no] = np.random.random()

        for _ in range(self._epochs):
            # Calculate function value
            f = self.coefficients['intercept']
            for key, c in self.coefficients.items():
                if key is not 'intercept':
                    f += c * self.features[int(key)]

            self.mse.append(self.cost_function(f, y, total=True))

            for key in self.coefficients:
                if key == 'intercept':
                    self.coefficients[key] -= self._learning_rate * np.mean(f-y)
                else:
                    self.coefficients[key] -= self._learning_rate * np.mean(np.multiply(X[int(key)], (f-y)))
