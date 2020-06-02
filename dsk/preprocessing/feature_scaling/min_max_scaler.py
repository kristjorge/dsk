import numpy as np


class MinMaxScaler:

    def __init__(self):
        self.scaled_features = {}

    def fit(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        for column in range(X.shape[1]):
            self.scaled_features[column] = ScaledFeature(min(X[:, column]), max(X[:, column]))

    def transform(self, X):
        self._check_dimensions(X)
        to_be_scaled = X.copy()
        for column in range(to_be_scaled.shape[1]):
            to_be_scaled[:, column] = self.scaled_features[column].transform(to_be_scaled[:, column])

        return to_be_scaled

    def inverse_transform(self, X):
        self._check_dimensions(X)
        scaled = X.copy()
        for column in range(scaled.shape[1]):
            scaled[:, column] = self.scaled_features[column].inverse_transform(scaled[:, column])

        return scaled

    @staticmethod
    def _check_dimensions(X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X


class ScaledFeature:

    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
        self.transform = np.vectorize(self._transform)
        self.inverse_transform = np.vectorize(self._inverse_transform)

    def _transform(self, value):
        return (value - self.minimum) / (self.maximum - self.minimum)

    def _inverse_transform(self, scaled_value):
        return scaled_value * (self.maximum - self.minimum) + self.minimum

