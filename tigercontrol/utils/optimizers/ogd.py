'''
OGD optimizer
'''
import jax.numpy as np
from tigercontrol.utils.optimizers.core import Optimizer
from tigercontrol.utils.optimizers.losses import mse
from tigercontrol import error

class OGD(Optimizer):
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
        self.T = 0

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
        self.T += 1

        # Make everything a list for generality
        is_list = True
        if(type(params) is not list):
            params = [params]
            grad = [grad]
            is_list = False
    
        lr = self.lr / np.sqrt(self.T)
        if self.max_norm:
            self.max_norm = np.maximum(self.max_norm, np.linalg.norm([np.linalg.norm(dw) for dw in grad]))
            lr = self.lr / self.max_norm
        new_params = [w - lr * dw for (w, dw) in zip(params, grad)]

        return new_params if is_list else new_params[0]


    def __str__(self):
        return "<OGD Optimizer, lr={}>".format(self.lr)



