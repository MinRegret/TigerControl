""" let's see if bandit optimization works! """

import sys
import os
import time

import jax 
import jax.numpy as np
import jax.random as random
from tigercontrol.utils import generate_key


class BanditMemory:
    def __init__(self, x_init, d, H, f=None, delta=0.1, initial_lr=0.1):
        """
        Description: implementation of the algorithm Online Bandit Optimization 
            with Memory.
        Args:
            x_init: the first value of x in K, convex set (not defineable here)
                note: is there any way to define K: the convex set?
            d: dimension of x_t (MUST BE SPECIFIED)
            H: memory of loss functions (MUST BE SPECIFIED)
            f: optional loss function fixed for all iterations
            delta: perturbation constant
            initial_lr: the initial learning rate (diminishes by T^(-3/4))
        """
        assert x_init.shape == (H, d), "x_init has invalid shape {}".format(x_init.shape)
        assert 0 < delta and delta < 1, "invalid delta"
        assert 0 < initial_lr, "invalid learning rate"
        self.x = x_init # x[0] is the oldest prediction, x[-1] the newest!
        self.d = d
        self.H = H
        self.f = f
        self.delta = delta
        self.initial_lr = initial_lr
        self.t = 0 # time

        #@jax.jit
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

        self.u = self._generate_uniform(self.d * self.H).reshape(H, d)

    def step(self, f_t=None): 
        """
        Description: perform one gradient estimate update step with respect to x, return loss
        Args:
            f_t: loss function with memory
        """
        if (f_t == None and self.f == None) or (f_t and self.f):
            raise Exception("loss function not properly specified!")
        self.t += 1

        # update u
        weird_norm = np.linalg.norm(self.u[:-1])
        new_u_norm = np.sqrt(1 - np.linalg.norm(self.u[1:])**2)
        self.u = self._update(self.u, self._generate_uniform(self.d, new_u_norm))

        # predict perturbed y = x + delta u
        delta_t = self.delta / self.t**0.25 # delta = O(T^(-1/4)) approximation
        y_t = self.x[-1] + delta_t * self.u[-1]
        y_t_hist = self.x + delta_t * self.u # history (y_t-H+1, ..., y_t)
        f = f_t if f_t else self.f
        loss_t = f(y_t_hist)

        # estimate gradient and perform update step
        g_t = (self.d * self.H / delta_t) * loss_t * np.sum(self.u, axis=0)
        lr = self.initial_lr / self.t**0.75 # eta_t = O(t^(-3/4))
        x_t = self.x[-1] - self.initial_lr * g_t
        self.x = self._update(self.x, x_t)

        return x_t, y_t, loss_t



