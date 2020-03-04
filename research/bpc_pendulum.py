# run BPC on cartpole

import jax
import jax.numpy as np
import numpy as onp
import time
import tigercontrol
from tigercontrol.environments import Environment

class Pendulum:
    def __init__(self, g=10.0):
        self.max_speed=20.
        #self.max_torque=1.
        self.max_torque=3. # INCREASED TORQUE
        self.dt=.05
        self.g = g
        self.action_space = (1,)
        self.observation_space = (2,)
        self.n, self.m = 2, 1

        @jax.jit
        def angle_normalize(x):
            x = np.where(x > np.pi, x - 2*np.pi, x)
            x = np.where(x < -np.pi, x + 2*np.pi, x)
            return x
        self.angle_normalize = angle_normalize

        @jax.jit
        def _dynamics(x, u):
            th, th_dot = x
            g = self.g
            m = 1.
            l = 1.
            dt = self.dt
            u = np.clip(u, -self.max_torque, self.max_torque)[0]
            th_dot_dot = (-3.*g)/(2.*l) * np.sin(th + np.pi) + 3./(m*l**2) * u
            new_th = self.angle_normalize(th + th_dot*dt)
            new_th_dot = th_dot + th_dot_dot*dt 
            new_th_dot = np.clip(new_th_dot, -self.max_speed, self.max_speed)
            return np.array([new_th, new_th_dot])
        self._dynamics = _dynamics
        jacobian = jax.jacrev(self._dynamics, argnums=(0,1))
        self.dynamics_jacobian = jax.jit(lambda x, u: jacobian(x, u))

    def get_dynamics(self, x, u):
        return self.dynamics_jacobian(x, u)

    def initialize(self):
        return self.reset()

    def step(self, u):
        self._state = self._dynamics(self._state, np.squeeze(u, axis=1))
        return np.expand_dims(self._state, axis=1)

    def reset(self):
        theta = 0. # random.uniform(generate_key(), minval=-np.pi, maxval=np.pi)
        thdot = np.array(onp.random.uniform(-1, 1))
        self._state = np.array([theta, thdot])
        return np.expand_dims(self._state, axis=1)


if __name__ == "__main__":
    env = Pendulum()
    x = env.reset()
    print("success! x = ", x)
    for i in range(10):
        u = np.zeros((1,1))
        x = env.step(u)
        print("x = ", x)

    a, b = env.get_dynamics(np.zeros(2), np.zeros(1))
    print("a: ", a)
    print("b: ", b)






