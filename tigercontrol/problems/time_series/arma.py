"""
Autoregressive moving-average
"""

import jax
import jax.numpy as np
import jax.random as random

import tigercontrol
from tigercontrol.utils import generate_key
from tigercontrol.problems.time_series import TimeSeriesProblem


class ARMA(TimeSeriesProblem):
    """
    Description: Simulates an autoregressive moving-average time-series.
    """

    compatibles = set(['ARMA-v0', 'TimeSeries'])

    def __init__(self):
        self.initialized = False
        self.has_regressors = False

    def initialize(self, p=3, q=3, noise_list = None, c=None, noise_magnitude=0.1, noise_distribution = 'normal'):
        """
        Description: Randomly initialize the hidden dynamics of the system.
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
            self.phi = 0.99 * phi / np.linalg.norm(phi, ord=1)
        else:
            self.phi = p
        if type(q) == int:
            self.psi = random.normal(generate_key(), shape=(q,))
        else:
            self.psi = q
        if(type(self.phi) is list):
            self.p = self.phi[0].shape[0]
        else:
            self.p = self.phi.shape[0]
        if(type(self.psi) is list):
            self.q = self.psi[0].shape[0]
        else:
            self.q = self.psi.shape[0]
        self.noise_magnitude, self.noise_distribution = noise_magnitude, noise_distribution
        self.c = random.normal(generate_key()) if c == None else c
        self.x = random.normal(generate_key(), shape=(self.p,))
        self.noise_list = None
        if(noise_list is not None):
            self.noise_list = noise_list
            self.noise = np.array(noise_list[0:self.q])
        elif(noise_distribution == 'normal'):
            self.noise = self.noise_magnitude * random.normal(generate_key(), shape=(self.q,))
        elif(noise_distribution == 'unif'):
            self.noise = self.noise_magnitude * random.uniform(generate_key(), shape=(self.q,), minval=-1., maxval=1.)

        def _step(x, noise, eps):

            if(type(self.phi) is list):
                x_ar = np.dot(x, self.phi[self.T])
            else:
                x_ar = np.dot(x, self.phi)

            if(type(self.psi) is list):
                x_ma = np.dot(noise, self.psi[self.T])
            else:
                x_ma = np.dot(noise, self.psi)

            x_new = self.c + x_ar + x_ma + eps

            next_x = np.roll(x, 1) 
            next_noise = np.roll(noise, 1)

            next_x = jax.ops.index_update(next_x, 0, x_new) # equivalent to self.x[0] = x_new
            next_noise = jax.ops.index_update(next_noise, 0, eps) # equivalent to self.noise[0] = eps  

            return (next_x, next_noise, x_new)

        self._step = jax.jit(_step)
        return self.x[0]

    def step(self):
        """
        Description: Moves the system dynamics one time-step forward.
        Args:
            None
        Returns:
            The next value in the ARMA time-series.
        """
        assert self.initialized
        self.T += 1
        if(self.noise_list is not None):
            self.x, self.noise, x_new = self._step(self.x, self.noise, self.noise_list[self.q + self.T - 1])
        else:
            if(self.noise_distribution == 'normal'):
                self.x, self.noise, x_new = self._step(self.x, self.noise, \
                    self.noise_magnitude * random.normal(generate_key()))
            elif(self.noise_distribution == 'unif'):
                self.x, self.noise, x_new = self._step(self.x, self.noise, \
                    self.noise_magnitude * random.uniform(generate_key(), minval=-1., maxval=1.))
        return x_new

    def hidden(self):
        """
        Description: Return the hidden state of the system.
        Args:
            None
        Returns:
            (x, eps): The hidden state consisting of the last p x-values and the last q
            noise-values.
        """
        assert self.initialized
        return (self.x, self.noise)

    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(ARMA_help)

    def __str__(self):
        return "<ARMA Problem>"


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


