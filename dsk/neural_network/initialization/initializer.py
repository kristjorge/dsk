import numpy as np
import math as m


class Initializer:

    def __init__(self, layer):
        self.layer = layer

    def init_w(self):
        raise NotImplementedError

    def init_b(self):
        return 0.01 * np.ones((self.layer.size, 1)).copy()


class RandomInitializer(Initializer):

    def __init__(self, layer):
        super().__init__(layer)

    def init_w(self):
        return np.random.normal(0, 1, (self.layer.size, self.layer.previous_layer.size)).copy()


class XavierInitializer(Initializer):

    def __init__(self, layer):
        super().__init__(layer)

    def init_w(self):
        w = np.random.normal(0, 1, (self.layer.size, self.layer.previous_layer.size))
        w *= 1 / m.sqrt(self.layer.previous_layer.size)
        return w.copy()

