import pandas as pd
import numpy as np
import dsk.preprocessing as preprocessing


def main():

    X = np.array([[1, 1], [2, 2], [4, 3], [4, np.nan], [5, 5], [np.nan, 6], [7, 7], [8, np.nan], [9, 9]])
    missing_values_preprocessor = preprocessing.MissingData(missing_values=np.nan, method='median')
    X[:, 0] = missing_values_preprocessor.transform(X[:, 0])
    X[:, 1] = missing_values_preprocessor.transform(X[:, 1])


if __name__ == '__main__':
    main()