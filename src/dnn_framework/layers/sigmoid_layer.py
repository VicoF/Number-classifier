from src.dnn_framework import Layer
import numpy as np


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        return 1 / (1 + np.exp(-x)), x

    def backward(self, output_grad, cache):
        y = 1 / (1 + np.exp(-cache))
        return (1 - y) * y * output_grad, {}
