"""
Non-PyBullet implementation of Planar Quadrotor
"""
import jax
import jax.numpy as np
import jax.random as random

import os
import tigercontrol
from tigercontrol.utils import generate_key, get_tigercontrol_dir
from tigercontrol.environments import Environment

# necessary for rendering
# from gym.envs.classic_control import rendering


class PlanarQuadrotor(Environment):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=9.81):
        self.initialized = False
        self.dt=.05
        self.m = 0.1 # kg
        self.L = 0.2 # m
        self.I = 0.004 # inertia, kg*m^2
        self.g = g
        self.viewer = None
        self.action_space = (2,)
        self.n = 6
        self.observation_space = (self.n,)

        @jax.jit
        def _dynamics(x, u):
            # state = [x, y, th, xdot, ydot, thdot]^T
            # u = [u1, u2]^T (propeller thrusts; u1: left propeller, u2: right propeller)
            state = x
            x, y, th, xdot, ydot, thdot = state
            u1, u2 = u
            g = self.g
            m = self.m 
            L = self.L
            I = self.I
            dt = self.dt
            xddot = -(u1+u2)*np.sin(th)/m # xddot
            yddot = (u1+u2)*np.cos(th)/m - g # yddot
            thddot = L*(u2 - u1)/I # thetaddot
            state_dot = np.array([xdot, ydot, thdot, xddot, yddot, thddot])

            new_state = state + state_dot*dt

            return new_state
        self._dynamics = dynamics

    def initialize(self):
        self.initialized = True
        return self.reset()

    def step(self,u):
        self.last_u = u
        state = self._dynamics(self.state, u)
        # costs = state[0]**2 + state[1]**2 + state[2]**2 + state[3]**2 + state[4]**2 + state[5]**2 + 0.1*u[0]**2 + 0.1*u[1]**2
        # costs = 0.0
        self.state = state
        return self.state # , -costs, False, {}

    def linearize_dynamics(self, x0, u0):
        # Linearize dynamics about x0, u0
        dyn_jacobian = jax.jit(jax.jacrev(self._dynamics, argnums=(0,1))) 
        F = dyn_jacobian(x0, u0)
        A = F[0]
        B = F[1]
        # F = np.hstack(dyn_jacobian(x0, u0)) 
        return A, B

    def reset(self):
        x = random.uniform(generate_key(), minval=-0.5, maxval=0.5)
        y = random.uniform(generate_key(), minval=-0.5, maxval=0.5)
        th = random.uniform(generate_key(), minval=-30*np.pi/180, maxval=30*np.pi/180)
        xdot = random.uniform(generate_key(), minval=-0.1, maxval=0.1)
        ydot = random.uniform(generate_key(), minval=-0.1, maxval=0.1)
        thdot = random.uniform(generate_key(), minval=-0.1, maxval=0.1)
        
        self.state = np.array([x, y, th, xdot, ydot, thdot])
        self.last_u = np.array([0.0, 0.0])
        return self.state 


    # def render(self, mode='human'):

    #     if self.viewer is None:
    #         self.viewer = rendering.Viewer(500,500)
    #         self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
    #         rod = rendering.make_capsule(1, .2)
    #         rod.set_color(.8, .3, .3)
    #         self.pole_transform = rendering.Transform()
    #         rod.add_attr(self.pole_transform)
    #         self.viewer.add_geom(rod)
    #         axle = rendering.make_circle(.05)
    #         axle.set_color(0,0,0)
    #         self.viewer.add_geom(axle)
    #         fname = os.path.join(get_tigercontrol_dir(), "environments/controller/assets/clockwise.png")
    #         self.img = rendering.Image(fname, 1., 1.)
    #         self.imgtrans = rendering.Transform()
    #         self.img.add_attr(self.imgtrans)

    #     self.viewer.add_onetime(self.img)
    #     self.pole_transform.set_rotation(self.state[0] + np.pi/2)
    #     if self.last_u:
    #         self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)
    #     return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None