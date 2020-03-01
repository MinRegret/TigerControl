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
        self.max_speed=20.
        self.max_torque=1.
        self.dt=.05
        self.g = g
        self.viewer = None
        self.action_space = (1,)
        self.observation_space = (2,)
        self.n, self.m = 2, 1
        self.rollout_controller = None


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
            # newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
            th_dot_dot = (-3.*g)/(2.*l) * np.sin(th + np.pi) + 3./(m*l**2) * u
            new_th = self.angle_normalize(th + th_dot*dt)
            new_th_dot = th_dot + th_dot_dot*dt 
            new_th_dot = np.clip(new_th_dot, -self.max_speed, self.max_speed)


            '''
            newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2) * u
            newth = self.angle_normalize(th + newthdot*dt)
            newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)'''
            return np.array([new_th, new_th_dot])
        self._dynamics = _dynamics

        # C_x, C_u = (np.diag(np.array([0.2, 0.05, 1.0, 0.05])), np.diag(np.array([0.05])))
        # self._loss = jax.jit(lambda x, u: x.T @ C_x @ x + u.T @ C_u @ u) # MUST store as self._loss

        # C_x, C_u = np.diag(np.array([0.1, 0.0, 0.0, 0.0])), np.diag(np.array([0.1]))
        # self._loss = jax.jit(lambda x, u: x.T @ C_x @ x + u.T @ C_u @ u)

        self._loss = jax.jit(lambda x,u : self.angle_normalize(x[0])**2 + .1*(u[0]**2))

        # stack the jacobians of environment dynamics gradient
        jacobian = jax.jacrev(self._dynamics, argnums=(0,1))
        self._dynamics_jacobian = jax.jit(lambda x, u: np.hstack(jacobian(x, u)))

        # stack the gradients of environment loss
        loss_grad = jax.grad(self._loss, argnums=(0,1))
        self._loss_grad = jax.jit(lambda x, u: np.hstack(loss_grad(x, u)))

        # block the hessian of environment loss
        block_hessian = lambda A: np.vstack([np.hstack([A[0][0], A[0][1]]), np.hstack([A[1][0], A[1][1]])])
        hessian = jax.hessian(self._loss, argnums=(0,1))
        self._loss_hessian = jax.jit(lambda x, u: block_hessian(hessian(x,u)))

        def _rollout(act, dyn, x_0, T):
            def f(x, i):
                u = act(x)
                x_next = dyn(x, u)
                return x_next, np.hstack((x, u))
            _, trajectory = jax.lax.scan(f, x_0, np.arange(T))
            return trajectory
        self._rollout = jax.jit(_rollout, static_argnums=(0,1,3))

    def rollout(self, controller, T, dynamics_grad=False, loss_grad=False, loss_hessian=False):
        # Description: Roll out trajectory of given baby_controller.
        if self.rollout_controller != controller: self.rollout_controller = controller
        x = self._state
        trajectory = self._rollout(controller.get_action, self._dynamics, x, T)
        transcript = {'x': trajectory[:,:self.n], 'u': trajectory[:,self.n:]}

        # optional derivatives
        if dynamics_grad: transcript['dynamics_grad'] = []
        if loss_grad: transcript['loss_grad'] = []
        if loss_hessian: transcript['loss_hessian'] = []
        for x, u in zip(transcript['x'], transcript['u']):
            if dynamics_grad: transcript['dynamics_grad'].append(self._dynamics_jacobian(x, u))
            if loss_grad: transcript['loss_grad'].append(self._loss_grad(x, u))
            if loss_hessian: transcript['loss_hessian'].append(self._loss_hessian(x, u))
        return transcript


    def initialize(self):
        self.initialized = True
        return self.reset()

    def step(self,u):
        self.last_u = np.clip(u, -self.max_torque, self.max_torque)[0] # for rendering
        state = self._dynamics(self._state, u)
        # costs = self.angle_normalize(state[0])**2 + .1*state[1]**2 + .001*(u**2)
        self._state = state
        return self._state, self._loss(state, u), False

    def reset(self):
        theta = random.uniform(generate_key(), minval=-np.pi, maxval=np.pi)
        thdot = random.uniform(generate_key(), minval=-1., maxval=1.)
        self._state = np.array([theta, thdot])
        self.last_u = 0.0
        return self._state


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
            fname = os.path.join(get_tigercontrol_dir(), "environments/control/images/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self._state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



