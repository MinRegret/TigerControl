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

# BPC definition
class BPC(Controller):
    def __init__(self, A, B, H = 3, HH = 3, lr = 0.001, delta = 0.1, include_bias = True, project = True):
        self.n, self.m = B.shape
        self.A, self.B = A, B

        self.lr, self.delta, self.H, self.HH = lr, delta, H, HH
        self.include_bias, self.project = include_bias, project

        self.t = 1 

        self.K, self.M, self.bias = LQR(A, B).K, np.zeros((H, self.m, self.n)), np.zeros((self.m, 1))

        # Past H + HH noises 
        self.w = np.zeros((H + HH, self.n, 1))

        # past state and past action
        self.x, self.u = np.zeros((self.n, 1)), np.zeros((self.m, 1))

        def _generate_uniform(shape, norm=1.00):
            v = random.normal(generate_key(), shape=shape)
            v = norm * v / np.linalg.norm(v)
            return v

        self._generate_uniform = _generate_uniform
        self.eps = self._generate_uniform((H+HH, H, self.m, self.n))
        self.eps_bias = self._generate_uniform((H+HH, self.m, 1))


    def update(self, cost):
        # 1. Get gradient estimates
        delta_M = cost * np.sum(self.eps, axis = 0)
        delta_bias = cost * np.sum(self.eps_bias, axis = 0)

        # 2 Execute updates
        self.M -= self.lr / self.t**0.75 * delta_M
        self.bias -= self.lr / self.t**0.75 * delta_bias

        # 3. Ensure norm is of correct size
        norm = np.linalg.norm(self.M)
        if(self.project and norm > (1-self.delta)):
            self.M *= (1-self.delta) / norm
            
        # 4. Get new epsilon for M
        self.eps = jax.ops.index_update(self.eps, 0, self._generate_uniform(
                    shape = (self.H, self.m, self.n), norm = np.sqrt(1 - np.linalg.norm(self.eps[1:])**2)))
        self.eps = np.roll(self.eps, -1, axis = 0)

        # 5. Get new epsilon for bias
        self.eps_bias = jax.ops.index_update(self.eps_bias, 0, self._generate_uniform(
                    shape = (self.m, 1), norm = np.sqrt(1 - np.linalg.norm(self.eps_bias[1:])**2)))
        self.eps_bias = np.roll(self.eps_bias, -1, axis = 0)

    def get_action(self, x):
        # 1. Get new noise
        self.w = jax.ops.index_update(self.w, 0, x - self.A @ self.x - self.B @ self.u)
        self.w = np.roll(self.w, -1, axis = 0)

        # 5. Update x
        self.x = x

        # 6. Update t
        self.t = self.t + 1

        # 3. Compute and return new action
        self.u = -self.K @ x + np.tensordot(self.M + self.delta * self.eps[-1], self.w[-self.H:], \
            axes = ([0, 2], [0, 1])) +  self.include_bias * (self.bias + self.delta * self.eps_bias[-1])

        return self.u