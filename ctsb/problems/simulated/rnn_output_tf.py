"""
--- DEPRECATED ---

Recurrent neural network output based on TensorFlow Keras implementation
"""
import jax
import jax.numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, SimpleRNN
import ctsb
from ctsb.utils.random import generate_key

class RNN_Output_TF(ctsb.Problem):
    """
    Produces outputs from a randomly initialized recurrent neural network.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, l=32, h=128, rnn=None):
        """
        Description:
            Randomly initialize the RNN.
        Args:
            n (int): Input dimension.
            m (int): Observation/output dimension.
            l (int): Default value 32. Length of RNN memory, i.e. only consider last l inputs when producing next output.
            h (int): Default value 128. Hidden dimension of RNN.
            rnn (model): Default value None. Pretrained RNN to replace the hidden dynamics (must still specify
                dimensions n and m), provided by the user.
        Returns:
            The first value in the time-series
        """
        self.T = 0
        self.initialized = True
        self.n, self.m, self.l, self.h = n, m, l, h

        if rnn == None:
            hidden = SimpleRNN(h, input_shape=(l,n))
            output = Dense(m)
            model = Sequential()
            model.add(hidden)
            model.add(output)
            model.compile(loss='mse', optimizer='sgd')
            hidden_model = Sequential()
            hidden_model.add(hidden)
            hidden_model.compile(loss='mse', optimizer='sgd')

            self.model = model
            self.hidden_model = hidden_model
        else:
            self.model = rnn
        self.x = np.zeros(shape=(l,n))

        y = self.model.predict(self.x.reshape(1, self.l, self.n))[0]
        return y
        
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
        self.x = jax.ops.index_update(self.x, jax.ops.index[1:,:], self.x[:-1,:]) # equivalent to self.x[1:,:] = self.x[:-1,:]
        self.x = jax.ops.index_update(self.x, jax.ops.index[0,:], x) # equivalent to self.x[0,:] = x

        y = self.model.predict(self.x.reshape(1, self.l, self.n))[0]
        return y

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
        return self.hidden_model.predict(self.x.reshape(1, self.l, self.n))[0]

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
        print(RNN_Output_help)



# string to print when calling help() method
RNN_Output_help = """

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
            l (int): Default value 32. Length of RNN memory, i.e. only consider last l inputs when producing next output.
            h (int): Default value 128. Hidden dimension of RNN.
            rnn (model): Default value None. Pretrained RNN to replace the hidden dynamics (must still specify
                dimensions n and m), provided by the user.
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







