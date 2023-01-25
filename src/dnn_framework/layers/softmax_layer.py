import numpy as np
from src.dnn_framework import Layer


def _jacobian(x):
    batch_size, n_classes = x.shape
    diag = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    diag[:, range(n_classes), range(n_classes)] = x
    x = x.reshape((batch_size, 1, n_classes))
    return diag - (np.transpose(x, (0, 2, 1)) @ x)


class Softmax(Layer):

    def forward(self, x):
        y = np.exp(x) / np.sum(np.exp(x), axis=1).reshape((x.shape[0], 1))
        return y, y

    def backward(self, output_grad, cache):
        x = cache
        batch_size, n_classes = x.shape
        return (_jacobian(x) @ output_grad.reshape((batch_size, n_classes, 1))).squeeze()

