"""
Long-short term memory output
"""
import jax
import jax.numpy as np
import jax.experimental.stax as stax
import ctsb
from ctsb.utils.random import generate_key

class LSTM_Output(ctsb.Problem):
    """
    Produces outputs from a randomly initialized recurrent neural network.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, h=64):
        """
        Description:
            Randomly initialize the RNN.
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
        self.W_hh = glorot_init(generate_key(), (4*h, h)) # maps h_t to gates
        self.W_xh = glorot_init(generate_key(), (4*h, n)) # maps x_t to gates
        self.b_h = np.zeros(4*h)
        jax.ops.index_update(self.b_h, jax.ops.index[h:2*h], np.ones(h)) # forget gate biased initialization
        self.W_out = glorot_init(generate_key(), (m, h)) # maps h_t to output
        self.cell = np.zeros(h) # long-term memory
        self.hid = np.zeros(h) # short-term memory
        self.sigmoid = lambda x: 1. / (1. + np.exp(-x)) # no JAX implementation of sigmoid it seems?
        return np.dot(self.W_out, self.hid)
        
    def step(self, x):
        """
        Description:
            Takes an input and produces the next output of the RNN.
        Args:
            x (numpy.ndarray): RNN input, an n-dimensional real-valued vector.
        Returns:
            The output of the RNN computed on the past l inputs, including the new x.
        """
        assert self.initialized
        assert x.shape == (self.n,)
        self.T += 1

        gate = np.dot(self.W_hh, self.hid) + np.dot(self.W_xh, x) + self.b_h 
        i, f, g, o = np.split(gate, 4) # order: input, forget, cell, output
        self.cell =  self.sigmoid(f) * self.cell + self.sigmoid(i) * np.tanh(g)
        self.hid = self.sigmoid(o) * np.tanh(self.cell)
        return np.dot(self.W_out, self.hid)

    def hidden(self):
        """
        Description:
            Return the hidden state of the RNN when computed on the last l inputs.
        Args:
            None
        Returns:
            h: The hidden state.
        """
        assert self.initialized
        return self.hid

    def close(self):
        """
        Not implemented
        """
        pass

    def help(self):
        """
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(LSTM_Output_help)



# string to print when calling help() method
LSTM_Output_help = """

-------------------- *** --------------------

Id: RNN-v0
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







