
"""
Gradient Pertubation Controller

written by Paula Gradu, Elad Hazan and Anirudha Majumdar 
"""

import jax.numpy as np
import tigercontrol
from tigercontrol.controllers import Controller
from jax import grad,jit
import jax.random as random
from tigercontrol.utils import generate_key
import jax
import scipy

class GPC(Controller):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    compatibles = set([])

    def __init__(self):
        self.initialized = False

    def _get_dims(self):
        try:
            n = self.B.shape[0] ## dimension of  the state x 
        except:
            n = 1
        try:
            m = self.B.shape[1] ## dimension of the control u
        except:
            m = 1
        return (n, m)

    def initialize(self, A, B, H = 3, HH = 30, K = None, x = None):
        """
        Description: Initialize the dynamics of the model
        Args:
            A,B (float/numpy.ndarray): system dynamics
            H (postive int): history of the controller 
            HH (positive int): history of the system 
            K (float/numpy.ndarray): Starting policy (optional). Defaults to LQR gain.
            x (float/numpy.ndarray): initial state (optional)
        """
        self.initialized = True

        self.A, self.B = A, B
        self.n, self.m = self._get_dims()
        self.H, self.HH = H, HH

        if(K is None):
            # solve the ricatti equation 
            X = scipy.linalg.solve_continuous_are(A, B, np.identity(self.n), np.identity(self.m))
            #compute LQR gain
            self.K = np.linalg.inv(B.T @ X @ B + np.identity(self.m)) @ (B.T @ X @ A)
        else:
            self.K = K

        self.x = np.zeros(self.n) if x is None else x
        self.u = np.zeros(self.m)

        ## internal parmeters to the class 
        self.T = 1 ## keep track of iterations, for the learning rate
        self.learning_rate = 1
        self.M = np.zeros((H, self.m, self.n))
        self.S = np.repeat(B.reshape(1, self.n, self.m), HH, axis=0) # previously [B for i in range(HH)]
        for i in range(1, HH):
            self.S = jax.ops.index_update(self.S, i, (A - B @ self.K) @ self.S[i-1]) 
        self.w_past = np.zeros((HH + H, self.n)) ## this are the previous perturbations, from most recent [0] to latest [HH-1]

        ## FUNCTIONS ##
        def _update_past(self_past, x):
            new_past = np.roll(self_past, self.n)
            new_past = jax.ops.index_update(new_past, 0, x)
            return new_past

        self._update_past = jit(_update_past)

        # This is the counterfactual loss function, we prefer not to differentiate it and use JAX 
        def the_complicated_loss_function(M, w_past):
            final = 0.0
            for i in range(self.HH):
                temp = np.tensordot(M, w_past[i:i+self.H], axes=([0,2],[0,1]))
                final = final + self.S[i] @ temp
            return np.sum(final ** 2)

        self.grad_fn = jit(grad(the_complicated_loss_function))  # compiled gradient evaluation function

        def _get_action(x_new, x_old, u_old, w_past, M, lr):
            w_new = x_new - np.dot(self.A, x_old) - np.dot(self.B, u_old)
            w_past_new = self._update_past(w_past, w_new)
            u_new = -self.K @ x_new + np.tensordot(M, w_past[:self.H], axes=([0,2],[0,1]))
            M_new = M - lr * self.grad_fn(M, w_past_new)
            return x_new, u_new, w_past_new, M_new

        self._get_action = jit(_get_action)

    def get_action(self, x):
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            x (float/numpy.ndarray): current state

        Returns:
            u(float/numpy.ndarray): action to take
        """

        self.T +=1

        lr = self.learning_rate / np.sqrt(self.T)
        self.x, self.u, self.w_past, self.M = self._get_action(x, self.x, self.u, self.w_past, self.M, lr)

        return self.u

    def plan(self, x, T):
        """
        Description: Plan next T actions.

        Args:
            x (float/numpy.ndarray): starting state
            T (int): number of timesteps to plan actions for
            
        Returns:
            u (list): list of actions to take
        """
        u = []
        for i in range(T):
            u.append(self.get_action(x))
            x = self.A @ x + self.B @ u[i]
        return u

    def __str__(self):
        return "<GPC Model>"

