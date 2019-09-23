"""
Linear dynamical system
"""

import jax
import jax.numpy as np
import jax.random as random

import tigercontrol
from tigercontrol.utils import generate_key
from tigercontrol.problems.control import ControlProblem


class LDS_Control(ControlProblem):
    """
    Description: Simulates a linear dynamical system.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, d, noise=1.0):
        """
        Description: Randomly initialize the hidden dynamics of the system.
        Args:
            n (int): Input dimension.
            m (int): Observation/output dimension.
            d (int): Hidden state dimension.
            noise (float): Default value 1.0. The magnitude of the noise (Gaussian) added
                to both the hidden state and the observable output.
        Returns:
            The first value in the time-series
        """
        self.initialized = True
        self.T = 0
        self.n, self.m, self.d, self.noise = n, m, d, noise

        # shrinks matrix M such that largest eigenvalue has magnitude k
        normalize = lambda M, k: k * M / np.linalg.norm(M, ord=2)

        # initialize matrix dynamics
        self.A = random.normal(generate_key(), shape=(d, d))
        self.B = random.normal(generate_key(), shape=(d, n))
        self.C = random.normal(generate_key(), shape=(m, d))
        self.D = random.normal(generate_key(), shape=(m, n))
        self.h = random.normal(generate_key(), shape=(d,))

        # adjust dynamics matrix A
        self.A = normalize(self.A, 1.0)
        self.B = normalize(self.B, 1.0)
        self.C = normalize(self.C, 1.0)
        self.D = normalize(self.D, 1.0)

        def _step(u, h, eps):
            eps_h, eps_y = eps
            next_h = np.dot(self.A, h) + np.dot(self.B, u) + self.noise * eps_h
            y = np.dot(self.C, next_h) + np.dot(self.D, u) + self.noise * eps_y
            return (next_h, y)

        self._step = jax.jit(_step)

        y = np.dot(self.C, self.h) + np.dot(self.D, np.zeros(n)) + noise * random.normal(generate_key(), shape=(m,))
        return y


    def step(self, u):
        """
        Description: Moves the system dynamics one time-step forward.
        Args:
            u (numpy.ndarray): control input, an n-dimensional real-valued vector.
        Returns:
            A new observation from the LDS.
        """
        assert self.initialized
        assert u.shape == (self.n,)
        self.T += 1

        self.h, y = self._step(u, self.h, (random.normal(generate_key(), shape=(self.d,)), random.normal(generate_key(), shape=(self.m,))))
        return y

    def hidden(self):
        """
        Description: Return the hidden state of the system.
        Args:
            None
        Returns:
            h: The hidden state of the LDS.
        """
        assert self.initialized
        return self.h

    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(LDS_Control_help)



# string to print when calling help() method
LDS_Control_help = """

-------------------- *** --------------------

Id: LDS-Control-v0
Description: Simulates a linear dynamical system.

Methods:

    initialize(n, m, d, noise=1.0)
        Description:
            Randomly initialize the hidden dynamics of the system.
        Args:
            n (int): Input dimension.
            m (int): Observation/output dimension.
            d (int): Hidden state dimension.
            noise (float): Default value 1.0. The magnitude of the noise (Gaussian) added
                to both the hidden state and the observable output.
        Returns:
            The first value in the time-series

    step(u)
        Description:
            Moves the system dynamics one time-step forward.
        Args:
            u (numpy.ndarray): control input, an n-dimensional real-valued vector.
        Returns:
            A new observation from the LDS.

    hidden()
        Description:
            Return the hidden state of the system.
        Args:
            None
        Returns:
            h: The hidden state of the LDS.

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""




