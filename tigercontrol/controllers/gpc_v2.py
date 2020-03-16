
"""
Gradient Pertubation Controller
"""

import tigercontrol
from tigercontrol.controllers import Controller
from tigercontrol.controllers import LQR
from tigercontrol.controllers.core import quad_loss, policy_loss

import jax
import jax.numpy as np
from jax import grad

# GPC definition
class GPC_v2(Controller):

    def __init__(self, env, K = None, H = 3, look_back = 3, cost_fn = quad_loss, lr = 0.001):
        """
        Description: Initialize the dynamics of the model
        Args:
            env (object): environment
            K (float/numpy.ndarray): Starting policy (optional). 
            H (postive int): history of the controller 
            look_back (positive int): history (rollout) of the system 
            cost_fn (function): cost function
            lr (float/numpy.ndarray): learning rate(s)
        """

        self.n, self.m = env.n, env.m # State & Action Dimensions
        self.env = env

        self.t = 1 # Time Counter (for decaying learning rate)
        self.lr, self.H, self.look_back = lr, H, look_back # Model Hyperparameters

        # Model Parameters: initial linear policy / perturbation contributions
        self.K = np.zeros((self.m, self.n)) if K is None else K
        self.params = np.zeros((H, self.m, self.n))

        # Past H + look_back noises
        self.w = np.zeros((H + look_back, self.n, 1))

        # past state and past action
        self.x, self.u = np.zeros((self.n, 1)), np.zeros((self.m, 1))

        self.determine_action = lambda params, x, w: -self.K @ x + \
                                        np.tensordot(params, w[-self.H:], axes = ([0, 2], [0, 1]))

        self.grad_policy = grad(policy_loss)

    def update_params(self, grad = None, cost_fn = None):
        """
        Description: Updates the parameters of the model
        Args:
            grad (float/numpy.ndarray): gradient of loss
            cost_fn (function): current loss function
            cost_val (float): current cost value
        """
        # 1. Update t
        self.t = self.t + 1 

        # 2. Get gradients if not provided
        delta_params = self.grad_policy(self.params, self.determine_action, \
            self.w, self.look_back, self.env, cost_fn) if grad is None else grad

        # 3. Execute parameter updates
        self.params -= self.lr * delta_params

    def update_history(self, x = None):
        """
        Description: Updates the system history tracked by of the model
        Args:
            x (float/numpy.ndarray): observed state
        """
        self.w = update_noise(self.w, x, self.u, self.env)
        self.x = x
        self.u = self.determine_action(self.params, x, self.w)

    def get_action(self, x):
        """
        Description: Return the action chosen by the controller for state x. No side-effects.
        Args:
            x (float/numpy.ndarray): system state
        """

        return self.determine_action(self.params, x, self.w)
