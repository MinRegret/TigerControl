'''
OGD optimizer
'''
from ctsb.models.optimizers.core import Optimizer
import jax
import jax.numpy as np
import jax.experimental.stax as stax

class OGD(Optimizer):
    """
    Description: Updates parameters based on correct value, loss and learning rate.
    Args:
        y (int/numpy.ndarray): True value at current time-step
        loss (function): specifies loss function to be used; defaults to MSE
        lr (float): specifies learning rate; defaults to 0.001.
    Returns:
        None
    """
    def __init__(self, pred, loss, learning_rate, params_dict):
        self.lr = learning_rate
        loss_fn = lambda model_params, a, b : loss(pred(model_params, a), b)
        self.grad_fn = jax.jit(jax.grad(loss_fn))
        self.t = params_dict['t']
        self.past = params_dict['past']
        self.max_norm = params_dict['max_norm']

    def update(self, params, x, y, metadata_dict = None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            y (int/numpy.ndarray): True value at current time-step
            loss (function): specifies loss function to be used; defaults to MSE
            lr (float): specifies learning rate; defaults to 0.001.
        Returns:
            None
        """
        grad = (np.dot(params, metadata_dict['past']) - y) * metadata_dict['past'] 
        if np.linalg.norm(grad) > self.max_norm:
            self.max_norm = np.linalg.norm(grad)

        self.t = self.t + 1
        self.lr = 1 / (self.max_norm * np.sqrt(self.t))
        params = params - self.lr * grad 
        return params
