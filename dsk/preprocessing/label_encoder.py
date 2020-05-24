import numpy as np


class LabelEncoder:

    def __init__(self, method='regular'):
        self.labels = []
        self.method = method
        self.y = None

    def fit(self, y):
        self.y = y

        distinct_label_counter = 0
        for label in self.y:
            if label not in self._all_distinct_labels:
                self.labels.append(Label(label, distinct_label_counter))
                distinct_label_counter += 1

    def transform(self):
        encoded = np.array([self._encode(label) for label in self.y])
        if self.method == 'normalized':
            for label in self.labels:
                label.encoded_value /= max(encoded)
                label.normalization_value = max(encoded)
            encoded = encoded / max(encoded)
        elif self.method == 'regular':
            pass
        else:
            quit('Not a valid encoding method')

        self.y = encoded
        return self.y

    def inverse_transform(self):
        self.y = np.array([self._decode(label) for label in self.y])
        return self.y

    def _encode(self, decoded_label):
        return [label.encoded_value for label in self.labels if label.decoded_value == decoded_label][0]

    def _decode(self, encoded_label):
        return [label.decoded_value for label in self.labels if label.encoded_value == encoded_label][0]

    @property
    def _all_distinct_labels(self):
        labels = []
        for label in self.labels:
            if label.decoded_value not in labels:
                labels.append(label.decoded_value)
        return labels


class Label:

    def __init__(self, decoded_value, encoded_value):
        self.decoded_value = decoded_value
        self.encoded_value = encoded_value
        self.normalization_value = None
