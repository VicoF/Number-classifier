from src.dnn_framework import Layer
import numpy as np


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        super().__init__()
        rng = np.random.default_rng()
        self.W = rng.normal(0, 2 / (output_count + input_count), size=(output_count, input_count))
        self.B = rng.normal(0, 2 / output_count, size=(output_count,))


    def get_parameters(self):
        return {"w": self.W, "b": self.B}

    def get_buffers(self):
        return {}

    def forward(self, x):
        return x @ self.W.T + self.B, x

    def backward(self, output_grad, cache):
        # (dL/dx) = (dL/Dy)@W
        input_grad = output_grad @ self.W
        # (dL/dW) = (dL/dY).T @ X
        w_grad = output_grad.T @ cache
        # (dL/dB) = (dL/dY)
        b_grad = output_grad.T @ np.ones(cache.shape[0])

        return input_grad, {"w": w_grad, "b": b_grad}
