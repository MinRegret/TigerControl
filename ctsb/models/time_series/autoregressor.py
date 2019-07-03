"""
AR(p): Linear combination of previous values
"""

import ctsb
import jax
import jax.numpy as np
from jax import grad, jit, vmap
from ctsb.models.time_series import TimeSeriesModel

class AutoRegressor(TimeSeriesModel):
    """
    Implements the equivalent of an AR(p) model - predicts a linear
    combination of the previous p observed values in a time-series
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, p):
        """
        Description:
            Initializes autoregressive model parameters
        Args:
            p (int): Length of history used for prediction
        """
        self.initialized = True

        self.past = np.zeros(p + 1)
        self.past = jax.ops.index_update(self.past, 0, 1)

        self.params = np.zeros(p + 1)
        self.params = jax.ops.index_update(self.params, p, 0) # default to LastValue (?) 0 -> 1

    def step(self, x):
        """
        Description:
            Run one timestep of the model in its environment then update internal parameters
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized

        self.past = jax.ops.index_update(self.past, jax.ops.index[1:self.past.shape[0] - 1], self.past[2:])
        self.past = jax.ops.index_update(self.past, self.past.shape[0] - 1, x)

        return np.dot(self.params, self.past)

    def predict(self, x):
        """
        Description:
            Predict next value given present value
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        
        assert self.initialized

        self.past = jax.ops.index_update(self.past, jax.ops.index[1:self.past.shape[0] - 1], self.past[2:])
        self.past = jax.ops.index_update(self.past, self.past.shape[0] - 1, x)

        return np.dot(self.params, self.past)

    def update(self, y, loss = None, lr = 0.001):
        """
        Description:
            Updates parameters based on correct value, loss and learning rate.
        Args:
            y (int/numpy.ndarray): True value at current time-step
            loss (function): specifies loss function to be used; defaults to MSE
            lr (float): specifies learning rate; defaults to 0.001.
        Returns:
            None
        """

        if(loss is None):
            # default to MSE
            def MSE(params, inputs, targets):
                preds = np.dot(params, inputs)
                return np.sum((preds - targets)**2)
            loss = MSE
        else:
            # check for correct format
            raise NotImplementedError

        f_grad = jit(grad(loss))
        #val_grad = 
        self.params = self.params - lr * f_grad(self.params, self.past, y)

        return

    def help(self):
        """
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(AutoRegressor_help)



# string to print when calling help() method
AutoRegressor_help = """

-------------------- *** --------------------

Id: AutoRegressor
Description: Implements the equivalent of an AR(p) model - predicts a linear
    combination of the previous p observed values in a time-series

Methods:

    initialize()
        Description:
            Initializes autoregressive model parameters
        Args:
            p (int): Length of history used for prediction

    step(x)
        Description:
            Run one timestep of the model in its environment then update internal parameters
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step

    predict(x)
        Description:
            Predict next value given present value
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step

    update(y, loss, lr)
        Description:
            Updates parameters based on correct value, loss and learning rate.
        Args:
            y (int/numpy.ndarray): True value at current time-step
            loss (function): (optional)
            lr (float):
        Returns:
            None

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""