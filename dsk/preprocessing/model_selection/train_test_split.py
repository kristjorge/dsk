import numpy as np
import math


def train_test_split(X, y, test_size=0.2):

    X_train = X.copy()
    X_test = X.copy()
    y_train = y.copy()
    y_test = y.copy()

    N_training_samples = int(math.floor(X.shape[0] * (1 - test_size)))
    N_test_samples = X.shape[0] - N_training_samples

    training_set_idx = list(range(0, X.shape[0]))
    test_set_idx = []

    for i in range(N_test_samples):
        new_test_set_idx = _sample_training_index(X.shape[0], test_set_idx)
        test_set_idx.append(new_test_set_idx)
        training_set_idx.remove(new_test_set_idx)

    test_set_idx.sort()
    training_set_idx.sort()
    X_train = np.delete(X_train, test_set_idx, axis=0)
    X_test = np.delete(X_test, training_set_idx, axis=0)
    y_train = np.delete(y_train, test_set_idx, axis=0)
    y_test = np.delete(y_test, training_set_idx, axis=0)

    return X_train, X_test, y_train, y_test


def _sample_training_index(max_bound, sampled_indices):
    new_idx = int(max_bound * np.random.random())
    if new_idx not in sampled_indices:
        pass
    else:
        new_idx = _sample_training_index(max_bound, sampled_indices)

    return new_idx