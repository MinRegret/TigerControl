"""
Autoregressive moving-average
"""

import jax
import jax.numpy as np
import jax.random as random

import ctsb
from ctsb.utils import generate_key


class ARMA(ctsb.Problem):
    """
    Simulates an autoregressive moving-average time-series.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, p, q, c=None):
        """
        Description:
            Randomly initialize the hidden dynamics of the system.
        Args:
            p (int/numpy.ndarray): Autoregressive dynamics. If type int then randomly
                initializes a Gaussian length-p vector with L1-norm bounded by 1.0. 
                If p is a 1-dimensional numpy.ndarray then uses it as dynamics vector.
            q (int/numpy.ndarray): Moving-average dynamics. If type int then randomly
                initializes a Gaussian length-q vector (no bound on norm). If p is a
                1-dimensional numpy.ndarray then uses it as dynamics vector.
            c (float): Default value follows a normal distribution. The ARMA dynamics 
                follows the equation x_t = c + AR-part + MA-part + noise, and thus tends 
                to be centered around mean c.
        Returns:
            The first value in the time-series
        """
        self.initialized = True
        self.T = 0
        if type(p) == int:
            phi = random.normal(generate_key(), shape=(p,))
            self.phi = 1.0 * phi / np.linalg.norm(phi, ord=1)
        else:
            assert len(p.shape) == 1
            self.phi = p
        if type(q) == int:
            self.psi = random.normal(generate_key(), shape=(q,))
        else:
            assert len(q.shape) == 1
            self.psi = q
        self.p = self.phi.shape[0]
        self.q = self.psi.shape[0]
        self.c = random.normal(generate_key()) if c == None else c
        self.x = random.normal(generate_key(), shape=(self.p,))
        self.noise = random.normal(generate_key(), shape=(q,))
        return self.x[0]

    def step(self):
        """
        Description:
            Moves the system dynamics one time-step forward.
        Args:
            None
        Returns:
            The next value in the ARMA time-series.
        """
        assert self.initialized
        self.T += 1
        x_ar = np.dot(self.x, self.phi)
        x_ma = np.dot(self.noise, self.psi)
        eps = random.normal(generate_key())
        x_new = self.c + x_ar + x_ma + eps
        for i in range(self.p-1, 0, -1): # equivalent to self.x[(1,:)] = self.x[(:,-1)]
            jax.ops.index_update(self.x, i, self.x[i-1])
        for i in range(self.q-1, 0, -1): # equivalent to self.noise[1:] = self.noise[:-1]
            jax.ops.index_update(self.noise, i, self.noise[i-1])
        jax.ops.index_update(self.x, 0, x_new) # equivalent to self.x[0] = x_new
        jax.ops.index_update(self.noise, 0, eps) # equivalent to self.noise[0] = eps        
        return x_new

    def hidden(self):
        """
        Description:
            Return the hidden state of the system.
        Args:
            None
        Returns:
            (x, eps): The hidden state consisting of the last p x-values and the last q
            noise-values.
        """
        assert self.initialized
        return (self.x, self.noise)

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
        print(ARMA_help)


# string to print when calling help() method
ARMA_help = """

-------------------- *** --------------------

Id: ARMA-v0
Description: Simulates an autoregressive moving-average time-series.

Methods:

    initialize(p, q, c=None)
        Description:
            Randomly initialize the hidden dynamics of the system.
        Args:
            p (int/numpy.ndarray): Autoregressive dynamics. If type int then randomly
                initializes a Gaussian length-p vector with L1-norm bounded by 1.0. 
                If p is a 1-dimensional numpy.ndarray then uses it as dynamics vector.
            q (int/numpy.ndarray): Moving-average dynamics. If type int then randomly
                initializes a Gaussian length-q vector (no bound on norm). If p is a
                1-dimensional numpy.ndarray then uses it as dynamics vector.
            c (float): Default value follows a normal distribution. The ARMA dynamics 
                follows the equation x_t = c + AR-part + MA-part + noise, and thus tends 
                to be centered around mean c.
        Returns:
            The first value in the time-series

    step()
        Description:
            Moves the system dynamics one time-step forward.
        Args:
            None
        Returns:
            The next value in the ARMA time-series.

    hidden()
        Description:
            Return the hidden state of the system.
        Args:
            None
        Returns:
            (x, eps): The hidden state consisting of the last p x-values and the last q
            noise-values.

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""


