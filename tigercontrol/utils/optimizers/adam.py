'''
Adam optimizer
'''

from tigercontrol.utils.optimizers.core import Optimizer
from tigercontrol.utils.optimizers.losses import mse
from tigercontrol import error
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
    def __init__(self, learning_rate=1.0, max_norm=True, beta_1=0.9, beta_2=0.999, eps=1e-7):
        self.lr = learning_rate
        self.max_norm = max_norm
        self.beta_1, self.beta_2 = beta_1, beta_2
        self.beta_1_t, self.beta_2_t = self.beta_1, self.beta_2
        self.eps = eps
        self.m, self.v = None, None

        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

        @jit # helper update controller
        def _update(params, grad, m, v, max_norm, beta_1_t, beta_2_t):
            new_m = [self.beta_1 * m_i + (1. - self.beta_1) * dw for (m_i, dw) in zip(m, grad)]
            new_v = [self.beta_2 * v_i + (1. - self.beta_2) * np.square(dw) for (v_i, dw) in zip(v, grad)]
            m_t = [m_i / (1 - beta_1_t) for m_i in new_m] # bias-corrected estimates
            v_t = [v_i / (1 - beta_2_t) for v_i in new_v]

            # maintain current power of betas
            beta_1_t, beta_2_t = beta_1_t * self.beta_1, beta_2_t * self.beta_2
            max_norm = np.where(max_norm, np.maximum(max_norm, np.linalg.norm([np.linalg.norm(dw) for dw in grad])), max_norm)
            lr = self.lr / np.where(max_norm, max_norm, 1.)
            new_params = [w - lr * m_i / (np.sqrt(v_i) + self.eps) for (w, v_i, m_i) in zip(params, v_t, m_t)]
            return new_params, new_m, new_v, max_norm, beta_1_t, beta_2_t
        self._update = _update

    def reset(self): # reset internal parameters
        self.beta_1_t, self.beta_2_t = self.beta_1, self.beta_2
        self.m, self.v = None, None

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
        if self.m == None: # initialize momentum and square grads
            self.m = [np.zeros(dw.shape) for dw in grad]
            self.v = [np.zeros(dw.shape) for dw in grad]

        updated_params = self._update(params, grad, self.m, self.v, self.max_norm, self.beta_1_t, self.beta_2_t)
        new_params, self.m, self.v, self.max_norm, self.beta_1_t, self.beta_2_t = updated_params
        return new_params if is_list else new_params[0]

    def __str__(self):
        return "<Adam Optimizer, lr={}>".format(self.lr)



