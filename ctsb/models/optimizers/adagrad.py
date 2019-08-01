'''
AdaGrad optimizer
'''
from ctsb.models.optimizers.core import Optimizer
from ctsb.models.optimizers.losses import mse
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
    def __init__(self, pred=None, loss=mse, learning_rate=1.0, hyperparameters={}):
        self.initialized = False
        self.lr = learning_rate
        self.hyperparameters = {'max_norm':10.0}
        self.hyperparameters.update(hyperparameters)
        self.max_norm = self.hyperparameters['max_norm']
        self.G = None
        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

    def set_predict(self, pred, loss=mse):
        """
        Description: Updates internally stored pred and loss functions
        Args:
            pred (function): predict function, must take params and x as input
            loss (function): loss function. defaults to mse.
        """
        self._is_valid_pred(pred, raise_error=True)
        _loss = lambda params, x, y: loss(pred(params=params, x=x), y)
        _custom_loss = lambda params, x, y, custom_loss: custom_loss(pred(params= params, x=x), y)
        self._grad = jit(grad(_loss))
        self._custom_grad = jit(grad(_custom_loss), static_argnums=[3])
        self.initialized = True

        # new code - initialize private array-wise gradient update method
        def _array_update(params, grad, G, lr):
            new_G = G + np.square(grad)
            new_params = params - lr * (grad / np.sqrt(new_G))
            return (new_params, new_G)
        self._array_update = jit(_array_update)

        def _list_update(params, grad, G, lr):
            new_vals = [self._array_update(w, dw, g, lr) for w, dw, g in zip(params, grad, G)]
            return ([w[0] for w in new_vals], [w[1] for w in new_vals])
        self._list_update = jit(_list_update)

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
        if self.G is None:
            if (type(params) is list):
                self.G = [1e-3 * np.ones(shape=w.shape) for w in params]
            else:
                self.G = 1e-3 * np.ones(shape=params.shape)

        grad = self.gradient(params, x, y, loss=loss) # defined in optimizers core class
        if (type(params) is list):
            new_params, self.G = self._list_update(params, grad, self.G, self.lr)
            norm = np.linalg.norm([np.linalg.norm(w) for w in new_params])
            if norm > self.max_norm:
                new_params = [w * self.max_norm / norm for w in new_params]
            return new_params

        new_params, self.G = self._array_update(params, grad, self.G, self.lr)
        norm = np.linalg.norm(new_params)
        if norm > self.max_norm:
            new_params = new_params * self.max_norm / norm
        return new_params



