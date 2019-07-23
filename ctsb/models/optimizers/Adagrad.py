
from ctsb.models.optimizers.core import Optimizer
import jax
import jax.numpy as np
import jax.experimental.stax as stax

class Adagrad(Optimizer):
    ''' Description: Adagrad optimizer '''
    def __init__(self, pred, loss, learning_rate, params_dict):
        self.lr = learning_rate
        loss_fn = lambda model_params, a, b : loss(pred(model_params, a), b)
        self.grad_fn = jax.jit(jax.grad(loss_fn))
        self.G = params_dict['G']
        self.past = params_dict['past']
        self.max_norm = params_dict['max_norm']
        self.order = params_dict['order']

    def update(self, params, x, y, metadata_dict=None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            y (int/numpy.ndarray): True value at current time-step
            params(list) : The model parameters
        Returns:
            updated params

        """
        grad = (np.dot(params, metadata_dict['past']) - y) * metadata_dict['past']
        if np.linalg.norm(grad) > self.max_norm:
            self.max_norm = np.linalg.norm(grad)

        # self.t = self.t + 1
        self.G = self.G + np.square(grad)
        # print(self.G)
        # print(grad / np.sqrt(self.G))
        # print('shalom')
        # print(self.params)
        params = params -  (grad / np.sqrt(self.G))
        if np.linalg.norm(params) > self.order:
            params = params / np.linalg.norm(self.params) 

        return params

