import numpy as np


class OneHotEncoder:

    def __init__(self):
        self.labels = []
        self.X = None

    def fit(self, X, columns, labels=None):
        self.X = X
        if labels is None:
            for c in columns:
                self.detect_labels(self.X[:, c], c)
        else:
            for idx, c in enumerate(columns):
                self.labels.append(Label(c, labels[idx]))

        # Set all encoded values to an np.array with size equal to the number of distinct values to encode
        label_columns = self._decoded_label_columns()
        for label_column in label_columns:
            for label_idx, label in enumerate(self._label_by_decoded_column(label_column)):
                label.encoded_label = np.zeros(self._num_distinct_labels_in_column(label_column), dtype=int)
                label.encoded_label[label_idx] = 1
                label.encoded_columns = range(label_column,
                                              label_column+self._num_distinct_labels_in_column(label_column)+1)

    def transform(self):
        label_columns = self._decoded_label_columns()
        for label_column in label_columns:
            new_cols = np.array([self._encode(label_column, label) for label in self.X[:, label_column]])
            self.X = np.column_stack((self.X[:, :label_column], new_cols, self.X[:, label_column:]))
            self.X = np.delete(self.X, new_cols.shape[1]+1, 1)

        return self.X

    def inverse_transform(self):
        label_columns = self._decoded_label_columns()
        for label_column in label_columns:
            decode_dimension = self._num_distinct_labels_in_column(label_column)
            new_cols = np.array([self._decode(label_column, vector) for vector in self.X[:, label_column:decode_dimension + 1]])
            self.X = np.delete(self.X, range(label_column, label_column+decode_dimension), axis=1)
            self.X = np.column_stack((self.X[:, :label_column], new_cols, self.X[:, label_column:]))
        return self.X

    def _encode(self, decoded_column, decoded_label):
        labels = self._label_by_decoded_column(decoded_column)
        return [label.encoded_label for label in labels if label.decoded_label == decoded_label][0]

    def _decode(self, decoded_column, encoded_label):
        labels = self._label_by_decoded_column(decoded_column)
        for label in labels:
            if (label.encoded_label == encoded_label).all():
                return label.decoded_label

    def detect_labels(self, X, c):
        # Loops over X and adds new labels to self.labels if it is not already in the list
        for value in X:
            if value not in [label.decoded_label for label in self.labels]:
                self.labels.append(Label(c, value))

    def _label_by_decoded_column(self, c):
        return [label for label in self.labels if label.decoded_column == c]

    def _decoded_label_columns(self):
        cols = []
        for label in self.labels:
            if label.decoded_column not in cols:
                cols.append(label.decoded_column)
        return cols

    def _num_distinct_labels_in_column(self, c):
        labels = self._label_by_decoded_column(c)
        return len(labels)


class Label:
    def __init__(self, column, decoded_label):
        self.decoded_column = column
        self.decoded_label = decoded_label
        self.encoded_columns = []
        self.encoded_label = None

