'''
SGD optimizer
'''
from ctsb.models.optimizers.core import Optimizer
import jax
import jax.numpy as np
import jax.experimental.stax as stax

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
    def __init__(self, pred, loss, learning_rate):
        self.lr = learning_rate
        self.pred = pred
        loss_fn = lambda model_params, a, b : loss(pred(model_params, a), b)
        self.grad_fn = jax.jit(jax.grad(loss_fn))

    def update(self, model_params, x, y_true):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            y (int/numpy.ndarray): True value at current time-step
            loss (function): specifies loss function to be used; defaults to MSE
            lr (float): specifies learning rate; defaults to 0.001.
        Returns:
            None
        """
        grad = self.grad_fn(model_params, x, y_true)
        
        if(type(model_params) is list):
            return [w - self.lr * dw for (w, dw) in zip(model_params, grad)]
        else:
            return model_params - self.lr * grad
        


