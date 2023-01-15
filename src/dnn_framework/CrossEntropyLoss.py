from src.dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        raise NotImplementedError()