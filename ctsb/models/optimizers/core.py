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
        self.initialized = False
        self.lr = learning_rate
        self.hyperparameters = hyperparameters
        self.pred = pred
        self.loss = loss
        if self._is_valid_pred(pred, raise_error=False) and self._is_valid_loss(loss, raise_error=False):
            self.set_predict(pred, loss=loss)

    def set_loss(self, new_loss):
        """ Description: updates internal loss """
        self.loss = new_loss
        if self._is_valid_pred(self.pred, raise_error=False):
            self.set_predict(self.pred, loss=self.loss)

    def set_predict(self, pred, loss=None):
        """
        Description: Updates internally stored pred and loss functions
        Args:
            pred (function): predict function, must take params and x as input
            loss (function): loss function. defaults to mse.
        """
        # check pred and loss input
        self._is_valid_pred(pred, raise_error=True)
        self.pred = pred
        if loss != None: self.loss = loss
        self._is_valid_loss(self.loss, raise_error=True)

        _loss = lambda params, x, y: self.loss(self.pred(params=params, x=x), y)
        _custom_loss = lambda params, x, y, custom_loss: custom_loss(pred(params=params, x=x), y)
        self._grad = jit(grad(_loss))
        self._custom_grad = jit(grad(_custom_loss), static_argnums=[3])
        self.initialized = True

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

    def _is_valid_loss(self, loss, raise_error=True):
        """ Description: checks that loss is a valid function to differentiate with respect to using jax """
        if not callable(loss):
            if raise_error:
                raise error.InvalidInput("Optimizer 'loss' input {} is not callable".format(loss))
            return False
        inputs = list(inspect.signature(loss).parameters)
        if len(inputs) != 2:
            if raise_error:
                raise error.InvalidInput("Optimizer 'loss' input {} must take two arguments as input".format(loss))
            return False
        try:
            jit_grad_loss = jit(grad(loss))
        except Exception as e:
            if raise_error:
                message = "JAX jit-grad failed on 'loss' input {}. Full error message: \n{}".format(loss, e)
                raise error.InvalidInput(message)
            return False
        return True

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



