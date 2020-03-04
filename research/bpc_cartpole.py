"""
Non-PyBullet implementation of CartPole
"""
import jax
import jax.numpy as np
import numpy.random as random

class CartPole:
    def __init__(self):
        self.initialized = False
        self.compiled = False
        self.rollout_controller = None
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        #self.theta_threshold_radians = 15 * 2 * np.pi / 360
        self.theta_threshold_radians = 10 * 2 * np.pi / 360
        self.x_threshold = 2.4

        self.action_space = (1,)
        self.observation_space = (4,)
        self.n, self.m = 4, 1

        # input to dynamics must have shape (n,) and (m,)!
        def _dynamics(x_0, u): # dynamics
            x, x_dot, theta, theta_dot = x_0 #np.split(x_0, 4)
            force = self.force_mag * np.clip(u, -1.0, 1.0)[0] # iLQR may struggle with clipping due to lack of gradient
            costh = np.cos(theta)
            sinth = np.sin(theta)
            temp = (force + self.polemass_length * theta_dot * theta_dot * sinth) / self.total_mass
            thetaacc = (self.gravity*sinth - costh*temp) / (self.length * (4.0/3.0 - self.masspole*costh*costh / self.total_mass))
            xacc  = temp - self.polemass_length * thetaacc * costh / self.total_mass
            x  = x + self.tau * x_dot # use euler integration by default
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            state = np.array([x, x_dot, theta, theta_dot])
            return state
        self._dynamics = _dynamics
        jacobian = jax.jacrev(self._dynamics, argnums=(0,1))
        self.dynamics_jacobian = jax.jit(lambda x, u: jacobian(x, u))

    def get_dynamics(self, x, u):
        return self.dynamics_jacobian(x, u)
        
    def reset(self):
        self._state = random.uniform(size=(4,), low=-0.05, high=0.05)
        #self._state = np.array([0.0, 0.03, 0.03, 0.03]) # reproducible results
        return np.expand_dims(self._state, axis=1)

    def step(self, action):
        action = np.squeeze(action, axis=1)
        self._state = self._dynamics(self._state, action)
        x, theta = self._state[0], self._state[2]
        x_lim, th_lim = self.x_threshold, self.theta_threshold_radians
        done = bool(x < -x_lim or x > x_lim or theta < -th_lim or theta > th_lim)
        return np.expand_dims(self._state, axis=1), done


if __name__ == "__main__":
    env = CartPole()
    x = env.reset()
    print("success! x = ", x)
    for i in range(10):
        u = np.zeros((1,1))
        x, done = env.step(u)
        print("x = ", x)




