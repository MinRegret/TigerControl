'''
Adam optimizer
'''

from ctsb.models.optimizers.core import Optimizer
from ctsb.models.optimizers.losses import mse
from jax import jit, grad
import jax.numpy as np

class Adam(Optimizer):
    """
    Description: Ordinary Gradient Descent optimizer.
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate
    Returns:
        None
    """
    def __init__(self, pred=None, loss=mse, learning_rate=1.0, hyperparameters={}):
        self.initialized = False
        self.lr = learning_rate

        self.hyperparameters = {'beta_1': 0.9, 'beta_2': 0.999, 'eps': 0.00000001}
        self.hyperparameters.update(hyperparameters)
        self.beta_1, self.beta_2 = self.hyperparameters['beta_1'], self.hyperparameters['beta_2']
        self.beta_1_t, self.beta_2_t = self.beta_1, self.beta_2
        self.eps = self.hyperparameters['eps']

        self.max_norm = 1.

        self.m, self.v = None, None

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

        if(self.m is None):
            if(type(params) is list):
                self.m = [np.zeros(dw.shape[0]) for dw in grad]
                self.v = [0 for dw in grad]
            else:    
                self.m = np.zeros(grad.shape[0])
                self.v = 0

        if(type(params) is list):
            self.m = [self.beta_1 * m + (1. - self.beta_1) * dw for (m, dw) in zip(self.m, grad)]
            self.v = [self.beta_2 * v + (1. - self.beta_2) * dw.T @ dw for (v, dw) in zip(self.v, grad)]
            # bias-corrected estimates
            m_t = [m / (1 - self.beta_1_t) for m in self.m]
            v_t = [v / (1 - self.beta_2_t) for v in self.v]
        else:
            self.m = self.beta_1 * self.m + (1. - self.beta_1) * grad
            self.v = self.beta_2 * self.v + (1. - self.beta_2) * grad.T @ grad
            # bias-corrected estimates
            m_t = self.m / (1 - self.beta_1_t)
            v_t = self.v / (1 - self.beta_2_t)

        # maintain current power of betas
        self.beta_1_t, self.beta_2_t = self.beta_1_t * self.beta_1, self.beta_2_t * self.beta_2

        if(type(params) is list):
            self.max_norm = np.maximum(self.max_norm, np.linalg.norm([np.linalg.norm(dw) for dw in grad]))
            lr = self.lr / self.max_norm
            return [w - lr / (np.sqrt(v) + self.eps) * m for (w, v, m) in zip(params, v_t, m_t)]
        else:
            self.max_norm = np.maximum(self.max_norm, np.linalg.norm(grad))
            lr = self.lr / self.max_norm
            return params - lr / (np.sqrt(v_t) + self.eps) * m_t
