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

    def initialize(self, n, m, d = None, noise=1.0, fully_observable = True):
        """
        Description: Randomly initialize the hidden dynamics of the system.
        Args:
            n (int): Observation/output dimension.
            m (int): control dimension.
            d (int): Hidden state dimension.
            noise (float): Default value 1.0. The magnitude of the noise (Gaussian) added
                to both the hidden state and the observable output.
        Returns:
            The first value in the time-series
        """
        self.initialized = True
        self.T = 0

        if(fully_observable):
            assert (n == d) or d is None, "If the system is fully observable, n must be equal to d."
        self.n, self.m, self.d, self.noise = n, m, d, noise
        self.fully_observable = fully_observable

        # shrinks matrix M such that largest eigenvalue has magnitude k
        normalize = lambda M, k: k * M / np.linalg.norm(M, ord=2)

        # initialize matrix dynamics
        self.A = random.normal(generate_key(), shape=(d, d))
        self.B = random.normal(generate_key(), shape=(d, m))
        self.C = random.normal(generate_key(), shape=(n, d))
        self.D = random.normal(generate_key(), shape=(n, m))
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

        y = np.dot(self.C, self.h) + np.dot(self.D, np.zeros(m)) + noise * random.normal(generate_key(), shape=(n,))
        
        if(fully_observable):
            return self.h

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
        # assert u.shape == (self.n,)
        self.T += 1

        self.h, y = self._step(u, self.h, (random.normal(generate_key(), shape=(self.d,)), random.normal(generate_key(), shape=(self.n,))))
        
        if(self.fully_observable):
            return self.h

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
