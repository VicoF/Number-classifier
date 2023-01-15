from src.dnn_framework.loss import Loss
import numpy as np

class MeanSquaredErrorLoss(Loss):
    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        cost = np.sum((x - target) ** 2)/x.size
        grad = (2 * (x - target))/x.size
        return cost, grad
