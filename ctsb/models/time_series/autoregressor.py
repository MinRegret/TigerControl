"""
AR(p): Linear combination of previous values
"""

import ctsb
import jax
import jax.numpy as np
from jax import grad, jit, vmap
from ctsb.models.time_series import TimeSeriesModel
from ctsb.models.optimizers.SGD import SGD
from ctsb.models.optimizers.losses import mse

class AutoRegressor(TimeSeriesModel):
    """
    Description: Implements the equivalent of an AR(p) model - predicts a linear
    combination of the previous p observed values in a time-series
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, p, optimizer = SGD, loss = mse, lr = 0.001):
        """
        Description: Initializes autoregressive model parameters
        Args:
            p (int): Length of history used for prediction
            optimizer (class): optimizer choice
            loss (class): loss choice
            lr (float): learning rate for update
        """
        self.initialized = True

        self.past = np.zeros(p + 1)
        self.past = jax.ops.index_update(self.past, 0, 1)

        self.params = np.zeros(p + 1)

        def _predict(params, inputs):
            return np.dot(params, inputs)
        self._predict = jax.jit(_predict)

        def _update_past(self_past, x):
            new_past = np.roll(self_past, -1)
            new_past = jax.ops.index_update(new_past, new_past.shape[0] - 1, x)
            return new_past
        self._update_past = jax.jit(_update_past)

        self.optimizer = optimizer(pred = self._predict, loss = loss, learning_rate = lr)

    def predict(self, x):
        """
        Description: Predict next value given present value
        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step
        """
        
        assert self.initialized

        self.past = self._update_past(self.past, x)

        return self._predict(self.params, self.past)

    def update(self, y):
        """
        Description: Updates parameters using the specified optimizer
        Args:
            y (int/numpy.ndarray): True value at current time-step
        Returns:
            None
        """

        self.params = self.optimizer.update(self.params, self.past, y)
        return

    def help(self):
        """
        Description: Prints information about this class and its methods.
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