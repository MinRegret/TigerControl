"""
Non-PyBullet implementation of Pendulum
"""
import jax
import jax.numpy as np
import jax.random as random

import os
import tigercontrol
from tigercontrol.utils import generate_key, get_tigercontrol_dir
from tigercontrol.environments import Environment

# necessary for rendering
from gym.envs.classic_control import rendering


class Pendulum(Environment):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0):
        self.initialized = False
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.viewer = None
        self.action_space = (1,)
        self.observation_space = (2,)

        @jax.jit
        def angle_normalize(x):
            x = np.where(x > np.pi, x - 2*np.pi, x)
            x = np.where(x < -np.pi, x + 2*np.pi, x)
            return x
        self.angle_normalize = angle_normalize

        @jax.jit
        def _dynamics(x, u):
            th, thdot = x
            g = self.g
            m = 1.
            l = 1.
            dt = self.dt
            u = np.clip(u, -self.max_torque, self.max_torque)[0]
            newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
            newth = self.angle_normalize(th + newthdot*dt)
            newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
            return np.array([newth, newthdot])
        self._dynamics = dynamics

    def initialize(self):
        self.initialized = True
        return self.reset()

    def step(self,u):
        self.last_u = np.clip(u, -self.max_torque, self.max_torque)[0] # for rendering
        state = self._dynamics(self.state, u)
        costs = self.angle_normalize(state[0])**2 + .1*state[1]**2 + .001*(u**2)
        self.state = state
        return self.state, -costs, False, {}

    def reset(self):
        theta = random.uniform(generate_key(), minval=-np.pi, maxval=np.pi)
        thdot = random.uniform(generate_key(), minval=-1., maxval=1.)
        self.state = np.array([theta, thdot])
        self.last_u = 0.0
        return self.state


    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = os.path.join(get_tigercontrol_dir(), "environments/controller/assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



