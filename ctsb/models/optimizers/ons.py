'''
Newton Step optimizer
'''

from ctsb.models.optimizers.core import Optimizer
from ctsb.models.optimizers.losses import mse
from jax import jit, grad
import jax.numpy as np


class ONS(Optimizer):
    """
    Online newton step algorithm.
    """

    def __init__(self, pred=None, loss=mse, learning_rate=1.0, hyperparameters={}):
        
        self.initialized = False
        
        self.lr = learning_rate
        self.max_norm = 1.
        
        self.hyperparameters = {'beta': 1., 'eps' : 0.125}
        self.hyperparameters.update(hyperparameters)
        self.beta, self.eps = self.hyperparameters['beta'], self.hyperparameters['eps']

        self.A = None

        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

    def update(self, params, x, y, loss=None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            params (list/numpy.ndarray): Parameters of model pred method
            x (float): input to model
            y (float): true label
            loss (function): loss function. defaults to input value.
        Returns:
            Updated parameters in same shape as input
        """
        assert self.initialized

        grad = self.gradient(params, x, y, loss=loss) # defined in optimizers core class

        if(self.A is None):
            self.A = np.eye(grad.shape[0]) * self.eps # initialize

        self.A += grad * grad.T # update

        self.max_norm = np.maximum(self.max_norm, np.linalg.norm(grad))
        lr = self.lr / self.max_norm # retained for normalization
        return params - lr * (1. / self.beta) * np.linalg.inv(self.A) @ grad



