from src.dnn_framework.losses.loss import Loss
from src.dnn_framework import Softmax
import numpy as np


class CrossEntropyLoss(Loss):

    softmax = Softmax()

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        x2, cache = self.softmax.forward(x)
        target = np.eye(x.shape[1])[target.T]
        loss = - np.sum(target * np.log(x2), axis=1)
        loss_grad = - target / x2
        input_grad = self.softmax.backward(loss_grad, cache)
        return np.mean(loss), input_grad/x.shape[0]
