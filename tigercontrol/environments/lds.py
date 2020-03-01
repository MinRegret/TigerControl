"""
Linear dynamical system base class
"""

import jax
import jax.numpy as np
import jax.random as random
import tigercontrol
from tigercontrol.utils import generate_key
from tigercontrol.environments import Environment


class LDS(Environment):
    """
    Description: The base, master LDS class that all other LDS subenvironments inherit. 
        Simulates a linear dynamical system with a lot of flexibility and variety in
        terms of hyperparameter choices.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, d = None, partially_observable = False, noise_distribution=None, \
        noise_magnitude=1.0, system_params = {}, initial_state = None):
        """
        Description: Randomly initialize the hidden dynamics of the system.
        Args:
            n (int): State dimension.
            m (int): control dimension.
            d (int): Observation dimension. (Note: MUST specify partially_observable=True for d to be used!)
            partially_observable (bool): whether to project x to y
            noise_distribution (None, string, func): if None then no noise. Valid strings include ['normal', 'uniform']. 
                Valid noise functions must map inputs n (x dim), x (state), u (action), w (previous noise), and t (current time)
                to either a scalar or an n-dimensional vector of real values.
            noise magnitude (float): magnitude of noise
            params (dict): specify A, B, C, and D matrices in system dynamics
            initial_state (None, vector): initial x. If None then randomly initialized
        Returns:
            The first value in the time-series
        """
        self.initialized = True
        self.T = 0
        self.n, self.m, self.d = n, m, d
        self.noise_magnitude = noise_magnitude
        params = system_params.copy() # avoid overwriting system_params input dict

        if d != None: # d may only be specified in a partially observable system
            assert partially_observable

        # random normal helper function
        gaussian = lambda dims: random.normal(generate_key(), shape=dims)

        # determine the noise function to use, allowing for conditioning on x, u, previous noise, and current t
        if (noise_distribution == None):           # case no noise
            if partially_observable:
                self.noise = lambda n, x, u, w, t: (0.0, 0.0)
            else:
                self.noise = lambda n, x, u, w, t: 0.0
        elif (noise_distribution == 'normal'):   # case normal distribution
            if partially_observable:
                self.noise = lambda n, x, u, w, t: (gaussian((n,)), gaussian((d,)))
            else:
                self.noise = lambda n, x, u, w, t: gaussian((n,))
        elif (noise_distribution == 'uniform'): # case uniform distribution
            if partially_observable:
                self.noise = lambda n, x, u, w, t: (random.uniform(generate_key(), shape=(n,), minval=-1, maxval=1), \
                    random.uniform(generate_key(), shape=(d,), minval=-1, maxval=1))
            else:
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

        # helper function that generates a random matrix with given dimensions
        for matrix, shape in {'A':(n, n), 'B':(n, m), 'C':(d, n), 'D':(d, m)}.items():
            if matrix not in params: 
                if (d == None) and (matrix == 'C' or matrix == 'D'): continue
                params[matrix] = gaussian(shape)
            else:
                assert params[matrix].shape == shape # check input has valid shape
        normalize = lambda M, k: k * M / np.linalg.norm(M, ord=2) # scale largest eigenvalue to k
        self.A = normalize(params['A'], 1.0)
        self.B = normalize(params['B'], 1.0)
        if partially_observable:
            self.C = normalize(params['C'], 1.0)
            self.D = normalize(params['D'], 1.0)

        # initial state
        self.x = gaussian((n,)) if initial_state is None else initial_state

        # different dynamics depending on whether the system is fully observable or not
        if partially_observable:
            def _step(x, u, eps):
                eps_x, eps_y = eps
                next_x = np.dot(self.A, x) + np.dot(self.B, u) + self.noise_magnitude * eps_x
                y = np.dot(self.C, next_x) + np.dot(self.D, u) + self.noise_magnitude * eps_y
                return (next_x, y)
            self._step = jax.jit(_step)
        else:
            def _step(x, u, eps):
                eps_x = eps
                next_x = np.dot(self.A, x) + np.dot(self.B, u) + self.noise_magnitude * eps_x
                return (next_x, next_x)
            self._step = jax.jit(_step)

        if partially_observable: # return partially observable state
            u, w = np.zeros(m), np.zeros(d)
            self.prev_noise = self.noise(n, self.x, u, w, self.T)
            y = np.dot(self.C, self.x) + np.dot(self.D, u) + self.noise_magnitude * self.prev_noise[1] # (state_noise, obs_noise)
            return y
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

    def hidden(self):
        """
        Description: Return the hidden state of the system.
        Args:
            None
        Returns:
            h: The hidden state of the LDS.
        """
        assert self.initialized
        return self.x
