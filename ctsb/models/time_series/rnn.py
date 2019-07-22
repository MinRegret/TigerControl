"""
Recurrent neural network output
"""

import jax
import jax.numpy as np
import jax.experimental.stax as stax
import ctsb
from ctsb.utils.random import generate_key
from ctsb.models.time_series import TimeSeriesModel
from ctsb.models.optimizers.SGD import SGD
from ctsb.models.optimizers.losses import mse

class RNN(TimeSeriesModel):
    """
    Produces outputs from a randomly initialized recurrent neural network.
    """

    compatibles = set(['TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.uses_regressors = True

    def initialize(self, n, m, l = 32, h = 64, optimizer = SGD, loss = mse, lr = 0.003):
        """
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

        # initialize jax.jitted predict and update functions
        def _fast_predict(params, x, hid):
            W_h, W_x, W_out, b_h = params
            next_hid = np.tanh(np.dot(W_h, hid) + np.dot(W_x, x) + b_h)
            y = np.dot(W_out, next_hid)
            return (y, next_hid)
        self._fast_predict = jax.jit(_fast_predict)

        def _slow_predict(params, x_list):
            W_h, W_x, W_out, b_h = params
            next_hid = np.zeros(self.h)
            for x in x_list:
                next_hid = np.tanh(np.dot(W_h, next_hid) + np.dot(W_x, x) + b_h)
            y = np.dot(W_out, next_hid)
            return y
        self._slow_predict = jax.jit(_slow_predict)

        def _update_x(self_x, x):
            new_x = np.roll(self_x, self.n)
            new_x = jax.ops.index_update(new_x, jax.ops.index[0,:], x)
            return new_x
        self._update_x = jax.jit(_update_x)

        self.optimizer = optimizer(pred = self._slow_predict, loss = loss, learning_rate = lr)

    def predict(self, x):
        """
        Description:
            Predict next value given observation
        Args:
            x (int/numpy.ndarray): Observation
        Returns:
            Predicted value for the next time-step
        """
        assert self.initialized

        self.x = self._update_x(self.x, x)
        y, self.hid = self._fast_predict(self.params, x, self.hid)

        return y

    def update(self, y):
        """
        Description:
            Updates parameters
        Args:
            y (int/numpy.ndarray): True value at current time-step
        Returns:
            None
        """
        self.params = self.optimizer.update(self.params, self.x, y)
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







