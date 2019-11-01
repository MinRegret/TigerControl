"""
ODE Shooting Controller
"""

import jax
import jax.numpy as np
import tigercontrol
from tigercontrol.controllers import Controller

class Shooting(Controller):
    """
    Description: Implements the shooting controller to solve second order boundary value
    environments with conditions y(0) = a and y(L) = b. Assumes that the
    second order BVP has been converted to a first order system of two
    equations.
    """
    # used to get trajectories from env.rollout
    class OpenLoopController(Controller):
        def reset(self, u):
            self.u = u
            self.t = 0
        def get_action(self, x):
            u = self.u[self.t]
            self.t += 1
            return u

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
        self.plan_cache = []

    def get_action(self, x, replan=False, horizon=None):
        if len(self.plan_cache) == 0 or replan:
            if horizon == None: horizon = self.T
            self.plan_cache = self.plan(x, horizon)
            u = self.plan_cache.pop(0)
        else:
            u = self.plan_cache.pop(0)
        return u

    def plan(self, x, T):
        control = self.OpenLoopController()
        u = [np.zeros(self.m,) for i in range(T)] # T copies of m-dimensional zero action vector
        for i in range(self.update_steps):
            control.reset(u)
            trajectory = self.env.rollout(control, T, dynamics_grad=True, loss_grad=True, loss_hessian=False)
            for t, dyn_t, grad_t in zip(range(T), trajectory['dynamics_grad'], trajectory['loss_grad']):
                dl_dx, dl_du, dx_du = grad_t[:self.n], grad_t[self.n:], dyn_t[:,self.n:]
                u_grad = dl_du + dl_dx @ dx_du
                u[t] = u[t] - self.lr * u_grad
        return u

    def __str__(self):
        return "<Shooting Controller>"


