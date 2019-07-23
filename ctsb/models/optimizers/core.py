"""
Core class for optimizers 
"""

import inspect
from jax import jit, grad
from ctsb.models.optimizers.losses import mse
from ctsb import error

class Optimizer():

    """
    Description: Core class for model optimizers
    Args:
        pred (function): a prediction function implemented with jax.numpy 
        loss (function): specifies loss function to be used; defaults to MSE
        learning_rate (float): learning rate. Default value 0.01
        hyperparameters (dict): additional optimizer hyperparameters
    Returns:
        None
    """
    def __init__(self, pred=None, loss=mse, learning_rate=0.01, hyperparameters={}):
        self.lr = learning_rate
        self.hyperparameters = hyperparameters
        if self._is_valid_pred(pred, raise_error=False):
            self.set_predict(pred, loss=loss)
        else:
            self.initialized = False

    def set_predict(self, pred, loss=mse):
        """
        Description: Updates internally stored pred and loss functions
        Args:
            pred (function): predict function, must take params and x as input
            loss (function): loss function. defaults to mse.
        """
        self._is_valid_pred(pred, raise_error=True)
        _loss = lambda params, x, y: loss(pred(params=params, x=x), y)
        _custom_loss = lambda params, x, y, custom_loss: custom_loss(pred(params=params, x=x), y)
        self._grad = jit(grad(_loss))
        self._custom_grad = jit(grad(_custom_loss), static_argnums=[3])
        self.initialized = True


    def update(self, params, x, y, loss=None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            params (list/numpy.ndarray): Parameters of model pred method
            x (float): input to model
            y (float): true label
            loss (function): loss function. defaults to mse.
        Returns:
            Updated parameters in same shape as input
        """
        assert self.initialized
        grad = self.gradient(params, x, y, loss=loss) # defined in optimizers core class
        if (type(params) is list):
            return [w - self.lr * dw for (w, dw) in zip(params, grad)]
        return params - self.lr * grad


    def gradient(self, params, x, y, loss=None):
        """
        Description: Updates parameters based on correct value, loss and learning rate.
        Args:
            params (list/numpy.ndarray): Parameters of model pred method
            x (float): input to model
            y (float): true label
        Returns:
            Gradient of parameters in same shape as input
        """
        assert self.initialized
        if loss:
            return self._custom_grad(params, x, y, loss)
        return self._grad(params, x, y)


    def _is_valid_pred(self, pred, raise_error=True):
        """ Description: checks that pred is a valid function to differentiate with respect to using jax """
        if not callable(pred):
            if raise_error:
                raise error.InvalidInput("Optimizer 'pred' input {} is not callable".format(pred))
            return False
        inputs = list(inspect.signature(pred).parameters)
        if 'x' not in inputs or 'params' not in inputs:
            if raise_error:
                raise error.InvalidInput("Optimizer 'pred' input {} must take variables named 'params' and 'x'".format(pred))
            return False
        try:
            grad_pred = grad(pred)
        except Exception as e:
            if raise_error:
                message = "JAX is unable to take gradient with respect to optimizer 'pred' input {}.\n".format(pred) + \
                "Please verify that input is implemented using JAX NumPy. Full error message: \n{}".format(e)
                raise error.InvalidInput(message)
            return False
        try:
            jit_grad_pred = jit(grad_pred)
        except Exception as e:
            if raise_error:
                message = "JAX jit optimization failed on 'pred' input {}. Full error message: \n{}".format(pred, e)
                raise error.InvalidInput(message)
            return False
        return True



