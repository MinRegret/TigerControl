"""
Bandit Optimizer with Memory and hyperparameter selection
""" 

import sys
import os
import time
import itertools

import jax 
import jax.numpy as np
import jax.random as random
import tigercontrol
from tigercontrol.utils import generate_key


class BanditMemoryHyperParam:
    def __init__(self, x_init, d, H, f, delta_list, initial_lr_list):
        """
        Description: implementation of the algorithm Online Bandit Optimization 
            with Memory, including hyperparameter selection!
        Args:
            x_init: the first value of x in K, convex set (not defineable here)
                note: is there any way to define K: the convex set?
            d: dimension of x_t (MUST BE SPECIFIED)
            H: memory of loss functions (MUST BE SPECIFIED)
            f: fixed loss function fixed for all iterations
            delta_list: list of perturbation constants to hyperparameter search through
            initial_lr: list of initial learning rates to search through
        """
        assert x_init.shape == (H, d), "x_init has invalid shape {}".format(x_init.shape)
        self.x = x_init # x[0] is the oldest prediction, x[-1] the newest!
        self.d = d
        self.H = H
        self.f = f

        self.delta_list = delta_list
        self.initial_lr_list = initial_lr_list

        @jax.jit
        def _update(v, v_new):
            v = np.roll(v, -self.d)
            v = jax.ops.index_update(v, -1, v_new)
            return v
        self._update = _update

        def _generate_uniform(d, norm=1.0):
            v = random.normal(generate_key(), shape=(d,))
            v = norm * v / np.linalg.norm(v)
            return v
        self._generate_uniform = _generate_uniform

    def step(self, t, x, u, delta, initial_lr):
        assert x.shape = (self.H, self.d)
        assert u.shape = (self.H, self.d)

        # update u
        new_u_norm = np.sqrt(1 - np.linalg.norm(u[1:])**2)
        u_new = self._update(u, self._generate_uniform(self.d, new_u_norm))

        # predict perturbed y = x + delta u
        delta_t = delta / (t+1)**0.25 # delta = O(T^(-1/4)) approximation
        y = x + delta_t * u # history (y_t-H+1, ..., y_t)
        loss_t = self.f(y)

        # estimate gradient and perform update step
        g_t = (self.d * self.H / delta_t) * loss_t * np.sum(u, axis=0)
        lr = self.initial_lr / (t+1)**0.75 # eta_t = O(t^(-3/4))
        x_t = self.x[-1] - lr * g_t
        #x_t = self.x[-1] - self.initial_lr * g_t # fixed learning rate
        x_new = self._update(x, x_t)
        return x_new, u_new, loss_t

    def run_trial(self, T_max, x_init, d, H, delta, initial_lr, stopping_rule=False):
        x = x_init
        u = self._generate_uniform(H*d).reshape(H, d)
        losses = []
        for t in range(T_max):
            x, u, l_t = self.step(t, x, u, delta, initial_lr)
            losses.append(l_t)
            if self._halting_rule(losses, l_t):
                break

    def _halting_rule(self, l, val, div=2): 
        """ 
        Description: return True if val is greater than median of list 
        Note: div can be set to gamma > 2 to make stopping rule stricter
        """
        if len(l) % 2 == 0:
            return val >= (l[int(len(l)/div)] + l[int(len(l)/div - 1)]) / 2
        return val >= l[int(len(l)/div)]




