"""
Long-short term memory output
"""
import jax
import jax.numpy as np
import jax.random as random
import jax.experimental.stax as stax
import tigercontrol
from tigercontrol.utils.random import generate_key
from tigercontrol.problems.control import ControlProblem

class LSTM_TimeSeries(ControlProblem):
    """
    Description: Produces outputs from a randomly initialized recurrent neural network.
    """
    compatibles = set(['TimeSeries'])
    
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
        self.has_regressors = True
        self.n, self.m, self.h = n, m, h

        glorot_init = stax.glorot() # returns a function that initializes weights
        self.W_hh = glorot_init(generate_key(), (4*h, h)) # maps h_t to gates
        self.W_xh = glorot_init(generate_key(), (4*h, n)) # maps x_t to gates
        self.b_h = np.zeros(4*h)
        self.b_h = jax.ops.index_update(self.b_h, jax.ops.index[h:2*h], np.ones(h)) # forget gate biased initialization
        self.W_out = glorot_init(generate_key(), (m, h)) # maps h_t to output
        self.cell = np.zeros(h) # long-term memory
        self.hid = np.zeros(h) # short-term memory

        def _step(x, hid, cell):
            sigmoid = lambda x: 1. / (1. + np.exp(-x)) # no JAX implementation of sigmoid it seems?
            gate = np.dot(self.W_hh, hid) + np.dot(self.W_xh, x) + self.b_h 
            i, f, g, o = np.split(gate, 4) # order: input, forget, cell, output
            next_cell =  sigmoid(f) * cell + sigmoid(i) * np.tanh(g)
            next_hid = sigmoid(o) * np.tanh(next_cell)
            y = np.dot(self.W_out, next_hid)
            return (next_hid, next_cell, y)

        self._step = jax.jit(_step)
        return self.step()
        
    def step(self):
        """
        Description: Takes an input and produces the next output of the RNN.
        Args:
            x (numpy.ndarray): RNN input, an n-dimensional real-valued vector.
        Returns:
            The output of the RNN computed on the past l inputs, including the new x.
        """
        assert self.initialized
        self.T += 1
        x = random.normal(generate_key(), shape=(self.n,))
        self.hid, self.cell, y = self._step(x, self.hid, self.cell)
        return x, y

    def hidden(self):
        """
        Description: Return the hidden state of the RNN when computed on the last l inputs.
        Args:
            None
        Returns:
            h: The hidden state.
        """
        assert self.initialized
        return (self.hid, self.cell)

    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(LSTM_TimeSeries_help)



# string to print when calling help() method
LSTM_TimeSeries_help = """

-------------------- *** --------------------

Id: LSTM-TimeSeries-v0
Description: Produces outputs from a randomly initialized recurrent neural network.

Methods:

    initialize(n, m, l=32, h=128, rnn=None)
        Description:
            Randomly initialize the RNN.
        Args:
            n (int): Input dimension.
            m (int): Observation/output dimension.
            h (int): Default value 64. Hidden dimension of RNN.
        Returns:
            The first value in the time-series

    step(x)
        Description:
            Takes an input and produces the next output of the RNN.
        Args:
            x (numpy.ndarray): RNN input, an n-dimensional real-valued vector.
        Returns:
            The output of the RNN computed on the past l inputs, including the new x.

    hidden()
        Description:
            Return the hidden state of the RNN when computed on the last l inputs.
        Args:
            None
        Returns:
            h: The hidden state.

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""







