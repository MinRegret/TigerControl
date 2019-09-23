"""
Recurrent neural network model
"""

import jax
import jax.numpy as np
import jax.experimental.stax as stax
import tigercontrol
from tigercontrol.utils.random import generate_key
from tigercontrol.models.time_series import TimeSeriesModel
from tigercontrol.models.optimizers import *
from tigercontrol.models.optimizers.losses import mse

class RNN(TimeSeriesModel):
    """
    Description: Produces outputs from a randomly initialized recurrent neural network.
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = True

    def initialize(self, n=1, m=1, l = 32, h = 64, optimizer = OGD, loss = mse, lr = 0.003):
        """
        Description: Randomly initialize the RNN.
        Args:
            n (int): Input dimension.
            m (int): Observation/output dimension.
            l (int): Length of memory for update step purposes.
            h (int): Default value 64. Hidden dimension of RNN.
            optimizer (class): optimizer choice
            loss (class): loss choice
            lr (float): learning rate for update
        """
        self.T = 0
        self.initialized = True
        self.n, self.m, self.l, self.h = n, m, l, h

        # initialize parameters
        glorot_init = stax.glorot() # returns a function that initializes weights
        W_h = glorot_init(generate_key(), (h, h))
        W_x = glorot_init(generate_key(), (h, n))
        W_out = glorot_init(generate_key(), (m, h))
        b_h = np.zeros(h)
        self.params = [W_h, W_x, W_out, b_h]
        self.hid = np.zeros(h)
        self.x = np.zeros((l, n))

        """ private helper methods"""
        @jax.jit
        def _update_x(self_x, x):
            new_x = np.roll(self_x, -self.n)
            new_x = jax.ops.index_update(new_x, jax.ops.index[-1,:], x)
            return new_x

        @jax.jit
        def _fast_predict(carry, x):
            params, hid = carry  # unroll tuple in carry
            W_h, W_x, W_out, b_h = params
            next_hid = np.tanh(np.dot(W_h, hid) + np.dot(W_x, x) + b_h)
            y = np.dot(W_out, next_hid)
            return (params, next_hid), y

        @jax.jit
        def _predict(params, x):
            _, y = jax.lax.scan(_fast_predict, (params, np.zeros(h)), x)
            return y[-1]

        self.transform = lambda x: float(x) if (self.m == 1) else x
        self._update_x = _update_x
        self._fast_predict = _fast_predict
        self._predict = _predict
        self._store_optimizer(optimizer, self._predict)

    def to_ndarray(self, x):
        """
        Description: If x is a scalar, transform it to a (1, 1) numpy.ndarray;
        otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
        Returns:
            A numpy.ndarray representation of x
        """
        x = np.asarray(x)
        if np.ndim(x) == 0:
            x = x[None]
        return x

    def predict(self, x, timeline = 1):
        """
        Description: Predict next value given observation
        Args:
            x (float/numpy.ndarray): Observation
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized
        self.x = self._update_x(self.x, self.to_ndarray(x))
        carry, y = self._fast_predict((self.params, self.hid), self.to_ndarray(x))
        _, self.hid = carry
        return y


    def forecast(self, x, timeline = 1):
        """
        Description: Forecast values 'timeline' timesteps in the future
        Args:
            x (float/numpy.ndarray):  Value at current time-step
            timeline (int): timeline for forecast
        Returns:
            Forecasted values 'timeline' timesteps in the future
        """
        assert self.initialized
        self.x = self._update_x(self.x, self.to_ndarray(x))
        carry, x = self._fast_predict((self.params, self.hid), self.to_ndarray(x))
        _, self.hid = carry
        hid = self.hid

        pred = [self.transform(x)]
        for t in range(timeline - 1):
            carry, x = self._fast_predict((self.params, hid), self.to_ndarray(x))
            _, self.hid = carry
            pred.append(self.transform(x))
        return pred

    def update(self, y):
        """
        Description: Updates parameters
        Args:
            y (int/numpy.ndarray): True value at current time-step
        Returns:
            None
        """
        self.params = self.optimizer.update(self.params, self.x, y)
        return

    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(RNN_help)

# string to print when calling help() method
RNN_help = """

-------------------- *** --------------------

Id: RNN
Description: Implements a Recurrent Neural Network model.

Methods:

    initialize(n, m, l = 32, h = 64, optimizer = SGD, loss = mse, lr = 0.003):
        Description:
            Randomly initialize the RNN.
        Args:
            n (int): Input dimension.
            m (int): Observation/output dimension.
            l (int): Length of memory for update step purposes.
            h (int): Default value 64. Hidden dimension of RNN.
            optimizer (class): optimizer choice
            loss (class): loss choice
            lr (float): learning rate for update

    predict(x)
        Description:
            Predict next value given observation
        Args:
            x (int/numpy.ndarray): Observation
        Returns:
            Predicted value for the next time-step

    update(y)
        Description:
            Updates parameters
        Args:
            y (int/numpy.ndarray): True value at current time-step
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







