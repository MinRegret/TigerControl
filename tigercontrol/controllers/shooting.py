"""
ODE Shooting Controller
"""

import jax
import jax.numpy as np
import tigercontrol
from tigercontrol.controllers import Controller

# used to get trajectories from env.rollout
class BabyController(Controller):
    def __init__(self, u):
        self.u = u
        self.t = 0
    def get_action(self, x):
        u = self.u[self.t]
        self.t += 1
        return u

class Shooting(Controller):
    """
    Description: Implements the shooting controller to solve second order boundary value
    environments with conditions y(0) = a and y(L) = b. Assumes that the
    second order BVP has been converted to a first order system of two
    equations.
    """
    compatibles = set([])

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, T, env, optimizer, update_steps=25, learning_rate=0.01):
        """
        Description: Initialize the dynamics of the controller.
        Args:
            n (int): observation dimension
            m (int): action dimension
            T (int): default timeline
            env (Environment): task on which to act on
            optimizer (Optimizer): optimizer that performs gradient descent
            update_steps (int): number of times to perform a gradient update step
        """
        self.n, self.m, self.T = n, m, T
        self.update_steps = update_steps
        self.env = env
        self.optimizer = optimizer
        self.lr = learning_rate # TEMPORARY, fix optimizers
        self.initialized = True

    def plan(self, x, T):
        u = np.zeros((T, self.m)) # T copies of m-dimensional zero action vector
        for _ in range(self.update_steps):
            trajectory = self.env.rollout(BabyController(u), T, dynamics_grad=True, loss_grad=True, loss_hessian=False)
            for t, dyn_t, grad_t in zip(range(T), trajectory['dynamics_grad'], trajectory['loss_grad']):
                dl_dx, dl_du, dx_du = grad_t[:self.n], grad_t[self.n:], dyn_t[:,self.n:]
                u_grad = dl_du + dl_dx @ dx_du
                u = jax.opt.index_add(u, t, -self.lr * u_grad)
        return u

    def __str__(self):
        return "<Shooting Controller>"


