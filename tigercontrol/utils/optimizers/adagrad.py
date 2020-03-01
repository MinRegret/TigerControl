'''
AdaGrad optimizer
'''
from tigercontrol.utils.optimizers.core import Optimizer
from tigercontrol.utils.optimizers.losses import mse
from tigercontrol import error
from jax import jit, grad
import jax.numpy as np


class Adagrad(Optimizer):
    """
    Description: Ordinary Gradient Descent optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate
    Returns:
        None
    """
    def __init__(self, learning_rate=1.0, max_norm=True):
        self.lr = learning_rate
        self.max_norm = max_norm
        self.G = None

        @jit
        def _update(params, grad, G, max_norm):
            new_G = [g + np.square(dw) for g, dw in zip(G, grad)]
            max_norm = np.where(max_norm, np.maximum(max_norm, np.linalg.norm([np.linalg.norm(dw) for dw in grad])), max_norm)
            lr = self.lr / np.where(max_norm, max_norm, 1.)
            new_params = [w - lr * dw / np.sqrt(g) for w, dw, g in zip(params, grad, new_G)]
            return new_params, new_G, max_norm
        self._update = _update

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
        
        # Make everything a list for generality
        is_list = True
        if(type(params) is not list):
            params = [params]
            grad = [grad]
            is_list = False
        if self.G == None: # first run
            self.G = [1e-3 * np.ones(shape=g.shape) for g in grad]

        new_params, self.G, self.max_norm = self._update(params, grad, self.G, self.max_norm)
        return new_params if is_list else new_params[0]

    def __str__(self):
        return "<AdaGrad Optimizer, lr={}>".format(self.lr)


