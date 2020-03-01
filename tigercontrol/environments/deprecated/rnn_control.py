"""
Recurrent neural network output
"""
import jax
import jax.numpy as np
import jax.experimental.stax as stax
import tigercontrol
from tigercontrol.utils.random import generate_key
from tigercontrol.environments import Environment

class RNN_Control(Environment):
    """
    Description: Produces outputs from a randomly initialized recurrent neural network.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, h=64):
        """
        Description: Randomly initialize the RNN.
        Args:
            n (int): Input dimension.
            m (int): Observation/output dimension.
            h (int): Default value 64. Hidden dimension of RNN.
        Returns:
            The first value in the time-series
        """
        self.T = 0
        self.initialized = True
        self.n, self.m, self.h = n, m, h

        glorot_init = stax.glorot() # returns a function that initializes weights
        self.W_h = glorot_init(generate_key(), (h, h))
        self.W_u = glorot_init(generate_key(), (h, n))
        self.W_out = glorot_init(generate_key(), (m, h))
        self.b_h = np.zeros(h)
        self.hid = np.zeros(h)

        self.rollout_controller = None
        self.target = jax.random.uniform(generate_key(), shape=(self.m,), minval=-1, maxval=1)

        '''
        def _step(x, hid):
            next_hid = np.tanh(np.dot(self.W_h, hid) + np.dot(self.W_x, x) + self.b_h)
            y = np.dot(self.W_out, next_hid)
            return (next_hid, y)'''

        def _dynamics(hid, u):
            next_hid = np.tanh(np.dot(self.W_h, hid) + np.dot(self.W_u, u) + self.b_h)
            y = np.dot(self.W_out, next_hid)
            return (next_hid, y)

        # self._step = jax.jit(_step)
        self._dynamics = jax.jit(_dynamics)
        
        
        self._loss = lambda x, u: (self.target - self._dynamics(x, u))**2

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
        return np.dot(self.W_out, self.hid)

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

    def step(self, u):
        """
        Description: Takes an input and produces the next output of the RNN.

        Args:
            x (numpy.ndarray): RNN input, an n-dimensional real-valued vector.
        Returns:
            The output of the RNN computed on the past l inputs, including the new x.
        """
        assert self.initialized
        assert u.shape == (self.n,)
        self.T += 1

        self.hid, y = self._dynamics(self.hid, u)
        return y

    def hidden(self):
        """
        Description: Return the hidden state of the RNN when computed on the last l inputs.
        Args:
            None
        Returns:
            h: The hidden state.
        """
        assert self.initialized
        return self.hid
