import numpy as np

from src.dnn_framework import Layer


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self.alpha = 0.1
        self.gamma = np.ones(input_count)
        self.beta = np.zeros(input_count)
        self.global_mean = np.zeros(input_count)
        self.global_variance = np.zeros(input_count)
        self.epsilon = 0.0000000001
        self.alpha = alpha

    def get_parameters(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def get_buffers(self):
        return {"global_mean": self.global_mean, "global_variance": self.global_variance}

    def forward(self, x):
        return self._forward_training(x) if self.is_training() else self._forward_evaluation(x)

    def backward(self, output_grad, cache):
        x, x_hat, mean, variance = cache
        x_hat_grad = output_grad * self.gamma
        var_grad = np.sum(x_hat_grad * (x - mean) * - 0.5 * (variance + self.epsilon) ** (-3 / 2), axis=0)
        mean_grad = -np.sum(x_hat_grad / np.sqrt(variance + self.epsilon), axis=0) - 2 / x.shape[0] * var_grad * np.sum(
            x - mean, axis=0)
        input_grad = x_hat_grad / np.sqrt(variance + self.epsilon) + 2 / x.shape[0] * var_grad * (
                    x - mean) + mean_grad / x.shape[0]
        gamma_grad = np.sum(output_grad * x_hat, axis = 0)
        beta_grad = np.sum(output_grad, axis=0)
        return input_grad, {"gamma": gamma_grad, "beta": beta_grad}

    def _forward_training(self, x):
        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0)
        self.global_mean = self._low_pass(self.global_mean, mean)
        self.global_variance = self._low_pass(self.global_variance, variance)
        x_hat = (x - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * x_hat + self.beta, (x, x_hat, mean, variance)

    def _forward_evaluation(self, x):
        x_hat = (x - self.global_mean) / np.sqrt(self.global_variance + self.epsilon)
        return self.gamma * x_hat + self.beta, x

    def _low_pass(self, x_prev, x):
        return (1 - self.alpha) * x_prev + self.alpha * x
