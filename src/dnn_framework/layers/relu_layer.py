from src.dnn_framework import Layer
import numpy as np


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        return np.where(x >= 0, x, 0), x

    def backward(self, output_grad, cache):
        return np.where(cache >= 0, 1, 0) * output_grad, {}
