import numpy as np


class OneHotEncoder:

    def __init__(self):
        self.columns_to_encode = {}

    def fit(self, X, columns, labels=None):
        self.columns_to_encode = {c: {} for c in columns}
        if labels is None:
            for c in columns:
                self.detect_labels(X[:, c], c)
        else:
            for idx, c in enumerate(columns):
                self.columns_to_encode[c][labels[idx]] = None

        # Set all encoded values to an np.array with size equal to the number of distinct values to encode
        for encode_col in self.columns_to_encode.values():
            label_counter = 0
            for encode_label_key in encode_col.keys():
                encode_col[encode_label_key] = np.zeros(len(encode_col), dtype=int)
                encode_col[encode_label_key][label_counter] = 1
                label_counter += 1

    def transform(self, X):
        for column_no, encode_col in self.columns_to_encode.items():
            new_cols = np.array([self.encode(column_no, label) for label in X[:, column_no]])
            X = np.column_stack((X[:,:column_no], new_cols, X[:, column_no:]))
            X = np.delete(X, new_cols.shape[1]+1, 1)
        return X

    def encode(self, column_no, input_label):
        encoded = self.columns_to_encode[column_no][input_label]
        return encoded

    def detect_labels(self, X, c):
        for value in X:
            if value not in self.columns_to_encode[c]:
                self.columns_to_encode[c][value] =  None

