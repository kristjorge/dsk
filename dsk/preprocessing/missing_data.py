import numpy as np


class MissingData:

    def __init__(self, missing_values=np.nan, method='mean'):
        self.missing_values = missing_values
        self.method = method

        if method == 'mean':
            self.replaced_by = self.find_mean
        elif method == 'most_frequent':
            self.replaced_by = self.find_most_frequent
        elif method == 'median':
            self.replaced_by = self.find_median
        elif method == 'constant':
            self.replaced_by = 'constant'
        else:
            self.replaced_by = self.find_mean

    def transform(self, X, y=None):
        X = X.copy()
        if len(X.shape) == 1:
            if self.method == 'constant':
                X = self.replace_value(X, y)
            else:
                X = self.replace_value(X, self.replaced_by(X))

        else:
            for idx in range(X .shape[1]):
                if self.method == 'constant':
                    X[:, idx] = self.replace_value(X[:, idx], y[idx])
                else:
                    X[:, idx] = self.replace_value(X[:, idx], self.replaced_by(X[:, idx]))

        return X

    def find_mean(self, X):
        x = self.filter_missing_values(X)
        return np.mean(x)

    def find_most_frequent(self, X):
        x = self.filter_missing_values(X)
        return np.bincount(x).argmax()

    def find_median(self, X):
        x = self.filter_missing_values(X)
        return np.median(x)

    def replace_value(self, X, value):
        X = np.array(X, dtype=np.float64)
        if np.isnan(self.missing_values):
            X = np.where(np.isnan(X), value, X)
        else:
            X = np.where(X == self.missing_values, value, X)

        return X

    def filter_missing_values(self, X):
        X = np.array(X, dtype=np.float64)
        if np.isnan(self.missing_values):
            return X[np.logical_not(np.isnan(X))]
        else:
            return np.delete(X, np.where(X == self.missing_values))
