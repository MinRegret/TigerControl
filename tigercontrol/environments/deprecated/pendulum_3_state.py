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

class Pendulum_3_State(Environment):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    """Constructs an InvertedPendulumDynamics model.
        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N m].
            max_bounds: Maximum bounds for action [N m].
            m: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                BatchAutoDiffDynamics constructor.
        Note:
            state: [sin(theta), cos(theta), theta']
            action: [torque]
            theta: 0 is pointing up and increasing counter-clockwise.
        """
    def __init__(self, 
                 dt=0.02, 
                 min_bounds=-7.25,
                 max_bounds=7.25,
                 m=1.0,
                 l=1.0,
                 g=9.80665):

        self.initialized = False
        # self.max_speed=8
        # self.max_torque=1.
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.dt=dt
        self.g = g
        self.viewer = None
        self.action_space = (1,)
        self.observation_space = (3,)
        self.n, self.m = 3, 1
        self.rollout_controller = None


        @jax.jit
        def angle_normalize(x):
            x = np.where(x > np.pi, x - 2*np.pi, x)
            x = np.where(x < -np.pi, x + 2*np.pi, x)
            return x
        self.angle_normalize = angle_normalize

        @jax.jit
        def _dynamics(x, u):
            # th, th_dot = x
            g = self.g
            m = 1.
            l = 1.
            dt = self.dt
            # u = np.clip(u, self.min_bounds, self.max_bounds)[0]
            u = self.tensor_constrain(u[0], self.min_bounds, self.max_bounds)

            # print("x = " + str(x))
            sin_th, cos_th, th_dot = x
            th = np.arctan2(sin_th, cos_th)
            # print("th = " + str(th))

            new_th = th + th_dot * dt

            th_dot_dot = (-3.0*g)/(2.0*l) * np.sin(th + np.pi) + 3.0/(m*(l**2)) * u
            new_th_dot = th_dot + th_dot_dot*dt 
            # print("np.sin(new_th): " + str(np.sin(new_th)))
            # print("np.cos(new_th): " + str(np.cos(new_th)))
            # print("new_th_dot : " + str(new_th_dot))
            return np.array((np.sin(new_th), np.cos(new_th), new_th_dot))
        self._dynamics = _dynamics

        # C_x, C_u = (np.diag(np.array([0.2, 0.05, 1.0, 0.05])), np.diag(np.array([0.05])))
        # self._loss = jax.jit(lambda x, u: x.T @ C_x @ x + u.T @ C_u @ u) # MUST store as self._loss

        # C_x, C_u = np.diag(np.array([0.1, 0.0, 0.0, 0.0])), np.diag(np.array([0.1]))
        # self._loss = jax.jit(lambda x, u: x.T @ C_x @ x + u.T @ C_u @ u)

        self._loss = jax.jit(lambda x,u : ((x[1] - 1.0))**2 + 0.1*(u[0]**2))

        self._terminal_loss = jax.jit(lambda x,u : 100*(((x[1] - 1.0)**2) + x[2]**2))

        # best setting: 4/10 -> loss: 0x[0], 1x[1], 0.1u[0], terminal: 100*, bounds: 
        # best setting: 4/10 -> loss: 0x[0], 1x[1], 0.1u[0], terminal: 100*x[1], bounds: 5
        # best setting: 7/10 -> loss: 0x[0], 1x[1], 0.1u[0], terminal: 0*x[0] + 100*rest, bounds: 7.25

        # stack the jacobians of environment dynamics gradient
        jacobian = jax.jacrev(self._dynamics, argnums=(0,1))
        self._dynamics_jacobian = jax.jit(lambda x, u: np.hstack(jacobian(x, u)))

        # stack the gradients of environment loss
        loss_grad = jax.grad(self._loss, argnums=(0,1))
        self._loss_grad = jax.jit(lambda x, u: np.hstack(loss_grad(x, u)))

        terminal_loss_grad = jax.grad(self._terminal_loss, argnums=(0,1))
        self._terminal_loss_grad = jax.jit(lambda x, u : np.hstack(terminal_loss_grad(x,u)))

        # block the hessian of environment loss
        block_hessian = lambda A: np.vstack([np.hstack([A[0][0], A[0][1]]), np.hstack([A[1][0], A[1][1]])])
        hessian = jax.hessian(self._loss, argnums=(0,1))
        self._loss_hessian = jax.jit(lambda x, u: block_hessian(hessian(x,u)))

        terminal_hessian = jax.hessian(self._terminal_loss, argnums=(0,1))
        self._terminal_loss_hessian = jax.jit(lambda x, u: block_hessian(terminal_hessian(x,u)))

        def _rollout(act, dyn, x_0, T):
            def f(x, i):
                u = act(x)
                x_next = dyn(x, u)
                return x_next, np.hstack((x, u))
            last_x, trajectory = jax.lax.scan(f, x_0, np.arange(T))
            return last_x, trajectory
        self._rollout = jax.jit(_rollout, static_argnums=(0,1,3))
        # self._rollout = _rollout

    def rollout(self, controller, T, dynamics_grad=False, loss_grad=False, loss_hessian=False):
        # Description: Roll out trajectory of given baby_controller.
        if self.rollout_controller != controller: self.rollout_controller = controller
        x = self._state
        # print("====================================")
        # print("x:" + str(x))
        last_x, trajectory = self._rollout(controller.get_action, self._dynamics, x, T)
        transcript = {'x': trajectory[:,:self.n], 'u': trajectory[:,self.n:]}
        transcript['x'] = np.append(transcript['x'], np.asarray([last_x]), axis=0)
        # print("LENGTHS OF X AND U RESPECTIVELY")
        # print(len(transcript['x']))
        # print(len(transcript['u']))
        # print(transcript['x'])
        # print(last_x)

        # optional derivatives
        if dynamics_grad: transcript['dynamics_grad'] = []
        if loss_grad: transcript['loss_grad'] = []
        if loss_hessian: transcript['loss_hessian'] = []
        for x, u in zip(transcript['x'][:-1], transcript['u']):
            if dynamics_grad: transcript['dynamics_grad'].append(self._dynamics_jacobian(x, u))
            if loss_grad: transcript['loss_grad'].append(self._loss_grad(x, u))
            if loss_hessian: transcript['loss_hessian'].append(self._loss_hessian(x, u))
        if loss_grad:
            transcript['loss_grad'][-1] = self._terminal_loss_grad(transcript['x'][-1], 0.0)
        if loss_hessian:
            '''
            print("transcript['loss_hessian'][-1]:")
            print(transcript['loss_hessian'][-1])
            print("transcript['x'][-1]:")
            print(transcript['x'][-1])
            print("self._terminal_loss_hessian(transcript['x'][-1], 0.0)")
            print(self._terminal_loss_hessian(transcript['x'][-1], np.array([0.0])))'''
            transcript['loss_hessian'][-1] = self._terminal_loss_hessian(transcript['x'][-1], np.array([0.0]))

        return transcript


    def initialize(self):
        self.initialized = True
        return self.reset()

    def step(self,u):
        self.last_u = self.tensor_constrain(u[0], self.min_bounds, self.max_bounds) # for rendering
        # print("u = " + str(u))
        state = self._dynamics(self._state, u)
        # costs = self.angle_normalize(state[0])**2 + .1*state[1]**2 + .001*(u**2)
        self._state = state
        return self._state, self._loss(state, u), False

    def reset(self):
        # theta = random.uniform(generate_key(), minval=-np.pi, maxval=np.pi)
        theta = np.pi
        # thdot = random.uniform(generate_key(), minval=self.min_bounds, maxval=self.max_bounds)
        thdot = 0
        self._state = np.array([np.sin(theta), np.cos(theta), thdot])
        self.last_u = thdot
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
        self.pole_transform.set_rotation(np.arctan2(self._state[0],self._state[1]) + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def tensor_constrain(self, u, min_bounds, max_bounds):
        """Constrains a control vector tensor variable between given bounds through
        a squashing function.
        This is implemented with Theano, so as to be auto-differentiable.
        Args:
            u: Control vector tensor variable [action_size].
            min_bounds: Minimum control bounds [action_size].
            max_bounds: Maximum control bounds [action_size].
        Returns:
            Constrained control vector tensor variable [action_size].
        """
        diff = (max_bounds - min_bounds) / 2.0
        mean = (max_bounds + min_bounds) / 2.0
        return diff * np.tanh(u) + mean

    def dynamics(self, x, u):
        return self._dynamics(x,u)



