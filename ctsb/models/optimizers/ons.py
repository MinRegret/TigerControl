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
        
        self.hyperparameters = {'beta': 20., 'eps' : 0.1}
        self.hyperparameters.update(hyperparameters)
        self.beta, self.eps = self.hyperparameters['beta'], self.hyperparameters['eps']

        self.A = None

        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

    def update(self, params, x, y, loss=mse):
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
            if(type(params) is list):
                self.A = [np.eye(dw.shape[0]) * self.eps for dw in grad]
            else:
                self.A = np.eye(grad.shape[0]) * self.eps # initialize

        if(type(params) is list):
            for i in range(len(self.A)):
                self.A[i] += grad[i] @ grad[i].T # update
        else:
            self.A += grad @ grad.T # update

        if (type(params) is list):
            self.max_norm = np.maximum(self.max_norm, np.linalg.norm([np.linalg.norm(dw) for dw in grad]))
            lr = self.lr / self.max_norm # retained for normalization
            return [w - lr * (1. / self.beta) * np.linalg.inv(A) @ dw for (w, A, dw) in zip(params, self.A, grad)]

        self.max_norm = np.maximum(self.max_norm, np.linalg.norm(grad))
        lr = self.lr / self.max_norm # retained for normalization
        return params - lr * (1. / self.beta) * np.linalg.inv(self.A) @ grad



