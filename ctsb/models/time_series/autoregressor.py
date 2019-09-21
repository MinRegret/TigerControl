"""
AR(p): Linear combination of previous values
"""

import ctsb
import jax
import jax.numpy as np
import jax.experimental.stax as stax
from ctsb.utils.random import generate_key
from ctsb.models.time_series import TimeSeriesModel
from ctsb.models.optimizers import *
from ctsb.models.optimizers.losses import mse

class AutoRegressor(TimeSeriesModel):
    """
    Description: Implements the equivalent of an AR(p) model - predicts a linear
    combination of the previous p observed values in a time-series
    """
    
    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = False

    def initialize(self, p = 3, optimizer = OGD):
        """
        Description: Initializes autoregressive model parameters

        Args:
            p (int): Length of history used for prediction
            optimizer (class): optimizer choice
            loss (class): loss choice
            lr (float): learning rate for update
        """
        self.initialized = True
        self.past = np.zeros(p)
        glorot_init = stax.glorot() # returns a function that initializes weights
        self.params = glorot_init(generate_key(), (p+1,1)).squeeze()

        def _update_past(self_past, x):
            new_past = np.roll(self_past, 1)
            new_past = jax.ops.index_update(new_past, 0, x)
            return new_past
        self._update_past = jax.jit(_update_past)

        def _predict(params, x):
            return np.dot(params, np.append(1.0, x))
        self._predict = jax.jit(_predict)

        self._store_optimizer(optimizer, self._predict)

    def predict(self, x):
        """
        Description: Predict next value given observation
        Args:
            x (int/numpy.ndarray): Observation
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized, "ERROR: Model not initialized!"

        if(type(x) is not float): x = x.squeeze()

        self.past = self._update_past(self.past, x) # squeeze to remove extra dimensions
        return self._predict(self.params, self.past)

    def forecast(self, x, timeline = 1):
        """
        Description: Forecast values 'timeline' timesteps in the future
        Args:
            x (int/numpy.ndarray):  Value at current time-step
            timeline (int): timeline for forecast
        Returns:
            Forecasted values 'timeline' timesteps in the future
        """
        assert self.initialized, "ERROR: Model not initialized!"

        self.past = self._update_past(self.past, x)
        past = self.past.copy()
        pred = []

        for t in range(timeline):
            x = self._predict(self.params, past)
            pred.append(x)
            past = self._update_past(past, x) 

        return np.array(pred)

    def update(self, y):
        """
        Description: Updates parameters using the specified optimizer
        Args:
            y (int/numpy.ndarray): True value at current time-step
        Returns:
            None
        """
        assert self.initialized, "ERROR: Model not initialized!"

        self.params = self.optimizer.update(self.params, self.past, y)

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
        Description: Initializes autoregressive model parameters

        Args:
            p (int): Length of history used for prediction
            optimizer (class): optimizer choice

    predict(x)
        Description: Predict next value given present value

        Args:
            x (int/numpy.ndarray):  Value at current time-step
        Returns:
            Predicted value for the next time-step

    update(y)
        Description: Updates parameters based on correct value.

        Args:
            y (int/numpy.ndarray): True value at current time-step
        Returns:
            None

    help()
        Description: Prints information about this class and its methods.

        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""