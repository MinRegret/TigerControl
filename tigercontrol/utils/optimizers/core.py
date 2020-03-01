"""
Core class for optimizers 
"""

import inspect
from jax import jit, grad
from tigercontrol.utils.optimizers.losses import mse
from tigercontrol import error

class Optimizer():
    """
    Description: Core class for controller optimizers
    
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate. Default value 0.01
        hyperparameters (dict): additional optimizer hyperparameters
    Returns:
        None
    """
    def __init__(self, learning_rate=1.0):
        self.lr = learning_rate

    def __str__(self):
        return "<Optimizer Core>"

    def __repr__(self):
        return self.__str__()


