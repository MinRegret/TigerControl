
"""
Gradient Pertubation Controller
"""

import jax.numpy as np
import numpy as onp
import tigercontrol
from tigercontrol.controllers import Controller
from jax import grad,jit
import jax.random as random
from tigercontrol.utils import generate_key
import jax
import scipy
from tigercontrol.controllers import LQR


quad = lambda x, u: np.sum(x.T @ x + u.T @ u)

# GPC definition
class GPC(Controller):
    def __init__(self, A, B, Q = None, R = None, cost_fn = quad, \
        H = 3, HH = 3, lr = 0.0001):
        """
        Description: Initialize the dynamics of the model
        Args:
            A,B (float/numpy.ndarray): system dynamics
            H (postive int): history of the controller
            HH (positive int): history of the system
            K (float/numpy.ndarray): Starting policy (optional). Defaults to LQR gain.
            x (float/numpy.ndarray): initial state (optional)
        """

        n, m = B.shape # State & Action Dimensions

        #### PERSONAL NOTE: if these are None, automatically do sys id!!!!
        self.A, self.B = A, B # System Dynamics

        self.t = 1 # Time Counter (for decaying learning rate)
        self.lr, self.H= lr, H # Model Hyperparameters

        # Model Parameters
        # initial linear policy / perturbation contributions / bias
        self.K, self.M = LQR(A, B, Q, R).K, np.zeros((H, m, n))

        # Past H + HH noises
        self.w = np.zeros((H + HH, n, 1))

        # past state and past action
        self.x, self.u = np.zeros((n, 1)), np.zeros((m, 1))

        # The Surrogate Cost Function
        def policy_loss(M, w, cost_t = cost_fn):
            y = np.zeros((n, 1))
            for h in range(HH - 1):
                v = -self.K @ y + np.tensordot(M, w[h : h + H], axes = ([0, 2], [0, 1]))
                y = A @ y + B @ v + w[h + H]
            # Don't update state at the end
            h += 1
            v = -self.K @ y + np.tensordot(M, w[h : h + H], axes = ([0, 2], [0, 1]))
            return cost_t(y, v)

        self.grad = grad(policy_loss)

        # If cost function stays the same throughout run, jit the gradient for efficiency
        if(cost_fn is not None):
            self.grad = jit(self.grad)

    def update(self, grad = None, cost = None):
        # 1. Get gradients if not provided
        if(grad == None):
            delta_M = self.grad(self.M, self.bias, self.w, cost)
        else:
            delta_M = grad

        # 2. Execute updates
        self.M -= self.lr * delta_M

    def get_action(self, x):
        # 1. Get new noise (will be located at w[-1])
        self.w = jax.ops.index_update(self.w, 0, x - self.A @ self.x - self.B @ self.u)
        self.w = np.roll(self.w, -1, axis = 0)

        # 2. Update x
        self.x = x

        # 3. Update t
        self.t = self.t + 1

        # 3. Compute and return new action
        self.u = -self.K @ x + np.tensordot(self.M, self.w[-self.H:], axes = ([0, 2], [0, 1]))

        return self.u
