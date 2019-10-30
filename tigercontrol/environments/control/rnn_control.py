"""
Recurrent neural network output
"""
import jax
import jax.numpy as np
import jax.experimental.stax as stax
import tigercontrol
from tigercontrol.utils.random import generate_key
from tigercontrol.environments import Environment

class RNN_Control(Environment):
    """
    Description: Produces outputs from a randomly initialized recurrent neural network.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, h=64):
        """
        Description: Randomly initialize the RNN.
        Args:
            n (int): Input dimension.
            m (int): Observation/output dimension.
            h (int): Default value 64. Hidden dimension of RNN.
        Returns:
            The first value in the time-series
        """
        self.T = 0
        self.initialized = True
        self.n, self.m, self.h = n, m, h

        glorot_init = stax.glorot() # returns a function that initializes weights
        self.W_h = glorot_init(generate_key(), (h, h))
        self.W_x = glorot_init(generate_key(), (h, n))
        self.W_out = glorot_init(generate_key(), (m, h))
        self.b_h = np.zeros(h)
        self.hid = np.zeros(h)

        def _step(x, hid):
            next_hid = np.tanh(np.dot(self.W_h, hid) + np.dot(self.W_x, x) + self.b_h)
            y = np.dot(self.W_out, next_hid)
            return (next_hid, y)

        self._step = jax.jit(_step)
        return np.dot(self.W_out, self.hid)
        
    def step(self, x):
        """
        Description: Takes an input and produces the next output of the RNN.

        Args:
            x (numpy.ndarray): RNN input, an n-dimensional real-valued vector.
        Returns:
            The output of the RNN computed on the past l inputs, including the new x.
        """
        assert self.initialized
        assert x.shape == (self.n,)
        self.T += 1

        self.hid, y = self._step(x, self.hid)
        return y

    def hidden(self):
        """
        Description: Return the hidden state of the RNN when computed on the last l inputs.
        Args:
            None
        Returns:
            h: The hidden state.
        """
        assert self.initialized
        return self.hid