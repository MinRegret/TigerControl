'''
SGD optimizer
'''
from tigercontrol.utils.optimizers.core import Optimizer
from tigercontrol.utils.optimizers.losses import mse

class SGD(Optimizer):
    """
    Description: Stochastic Gradient Descent optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate
    Returns:
        None
    """
    def __init__(self, learning_rate=0.0001):
        self.lr = learning_rate

    def update(self, params, grad):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            params (list/numpy.ndarray): Parameters of controller pred controller
            x (float): input to controller
            y (float): true label
            loss (function): loss function. defaults to input value.
        Returns:
            Updated parameters in same shape as input
        """
        if (type(params) is list):
            return [w - self.lr * dw for (w, dw) in zip(params, grad)]
        return params - self.lr * grad

    def __str__(self):
        return "<SGD Optimizer, lr={}>".format(self.lr)