import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import numpy as onp
import jax.random as random
import tigercontrol
from tigercontrol.utils.random import set_key, generate_key
from tigercontrol.environments import Environment
from tigercontrol.controllers import Controller
from jax import grad,jit

"""
Linear dynamical system
"""


class LDS(Environment):
    """
    Description: The base, master LDS class that all other LDS subenvironments inherit. 
        Simulates a linear dynamical system with a lot of flexibility and variety in
        terms of hyperparameter choices.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, noise_distribution=None, noise_magnitude=1.0, A = None, B = None, initial_state = None):
        """
        Description: Randomly initialize the hidden dynamics of the system.
        Args:
            n (int): State dimension.
            m (int): control dimension.
            noise_distribution (None, string, func): if None then no noise. Valid strings include ['normal', 'uniform']. 
                Valid noise functions must map inputs n (x dim), x (state), u (action), w (previous noise), and t (current time)
                to either a scalar or an n-dimensional vector of real values.
            noise magnitude (float): magnitude of noise
            params (dict): specify A, B
            initial_state (None, vector): initial x. If None then randomly initialized
        Returns:
            The first value in the time-series
        """
        self.initialized = True
        self.T = 0
        self.n, self.m = n, m
        self.noise_magnitude = noise_magnitude

        # random normal helper function
        gaussian = lambda dims: random.normal(generate_key(), shape=dims)
        
        self.A = gaussian((n, n))
        self.B = gaussian((n, m))
        
        self.A /= np.linalg.norm(self.A)
        self.B /= np.linalg.norm(self.B)

        # determine the noise function to use, allowing for conditioning on x, u, previous noise, and current t
        if (noise_distribution == None):           # case no noise
            self.noise = lambda n, x, u, w, t: 0.0
        elif (noise_distribution == 'normal'):   # case normal distribution
            self.noise = lambda n, x, u, w, t: gaussian((n,))
        elif (noise_distribution == 'uniform'): # case uniform distribution
            self.noise = lambda n, x, u, w, t: random.uniform(generate_key(), shape=(n,), minval=-1, maxval=1)
        else:                                      # case custom function
            assert callable(noise_distribution), "noise_distribution not valid input" # assert input is callable
            from inspect import getargspec
            arg_sub = getargspec(noise_distribution).args # retrieve all parameters taken by provided function
            for arg in arg_sub:
                assert arg in ['n', 'x', 'u', 'w', 't'], "noise_distribution takes invalid input"
            def noise(n, x, u, w, t):
                noise_args = {'n': n, 'x': x, 'u': u, 'w': w, 't': t}
                arg_dict = {k:v for k,v in noise_args.items() if k in arg_sub}
                return noise_distribution(**arg_dict)
            self.noise = noise

        # initial state
        #self.x = np.zeros(n) if initial_state is None else initial_state
        self.x = gaussian((n,)) if initial_state is None else initial_state
        
        def _step(x, u, eps):
            eps_x = eps
            next_x = np.dot(self.A, x) + np.dot(self.B, u) + self.noise_magnitude * eps_x
            return (next_x, next_x)
        self._step = jax.jit(_step)

        self.prev_noise = np.zeros(n)
        return self.x # return true current state


    def step(self, u):
        """
        Description: Moves the system dynamics one time-step forward.
        Args:
            u (numpy.ndarray): control input, an n-dimensional real-valued vector.
        Returns:
            A new observation from the LDS.
        """
        assert self.initialized
        self.T += 1
        self.prev_noise = self.noise(self.n, self.x, u, self.prev_noise, self.T)
        self.x, y = self._step(self.x, u, self.prev_noise)
        return y # even in fully observable case, y = self.x

        