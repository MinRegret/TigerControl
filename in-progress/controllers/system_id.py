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

class SystemID(Controller):
    """
    Description: Identifies unknown matrices A, B of linear dynamical system.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, K, k=None, T_0=None):
        """
        Description: Initialize the dynamics of the model
        Args:
            n, m (int): system dynamics dimension
            K (np.array): control matrix
            k (int): history to use when estimating A, B
            T_0 (int): number of total iterations (recommend T_0 = O(T^(2/3)))
        """
        self.initialized = True
        self.n, self.m = n, m
        self.T = 0
        self.K = K
        self.k = k
        self.T_0 = T_0 # our implementation uses T instead of T_0
        self.x_history = [] # history of x's
        self.eta = [] # perturbations


    def get_action(self, x_t):
        """ return action """
        eta_t = 1 - 2*random.randint(generate_key(), minval=0, maxval=2, shape=(self.m,))
        self.eta.append(eta_t)
        return - np.dot(self.K, x_t) + eta_t

    def update(self, x_new):
        """ takes new state and updates internal memory """
        self.T += 1
        self.x_history.append(x_new)
        return self.u

    def system_id(self):
        """ returns current estimate of hidden system dynamics """
        assert self.T > 0
        k = self.k if self.k else 0.1 * self.T
        eta_np = np.array(self.eta) # turn into np array
        x_np = np.array(self.x_history)
        scan_len = self.T-k-1 # need extra -1 because we iterate over j=0,..,k
        N_j = np.array([np.dot(x_np[j+1:j+1+scan_len], eta_np[j:j+scan_len]) for j in range(k+1)]) / scan_len
        C_0, C_1 = N_j[:k], N_j[1:k+1]
        B = N_j[0]
        A = C_1 @ C_0.T @ np.linalg.inv(C_0 @ C_0.T) + B @ self.K
        return (A, B)


        