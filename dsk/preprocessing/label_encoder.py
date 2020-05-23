import numpy as np


class LabelEncoder:

    def __init__(self, method='regular'):
        self.labels = []
        self.method = method

    def fit(self, y):
        if len(y.shape) == 1:
            y = y.reshape(-1)

        distinct_label_counter = 0
        for label in y[:, 0]:
            if label not in self.labels:
                self.labels.append(Label(label, distinct_label_counter))
                distinct_label_counter += 1

    # def transform

class Label:

    def __init__(self, decoded_value, encoded_value):
        self.decoded_value = decoded_value
        self.encoded_value = encoded_value