"""
Long-short term memory output
"""
import jax
import jax.numpy as np
import jax.experimental.stax as stax
import tigercontrol
from tigercontrol.utils.random import generate_key
from tigercontrol.environments import Environment

class LSTM_Control(Environment):
    """
    Description: Produces outputs from a randomly initialized recurrent neural network.
    """

    def __init__(self):
        self.initialized = False

    # def initialize(self, n, m, h=64):
    def initialize(self, u_dim, y_dim, hid_dim=64):
        """
        Description: Randomly initialize the RNN.
        Args:
            u_dim (int): Input dimension.
            y_dim (int): Observation/output dimension.
            hid_dim (int): Default value 64. Hidden dimension of RNN.
        Returns:
            The first value in the time-series
        """

        self.T = 0
        self.initialized = True
        # self.n, self.m, self.h = n, m, h

        self.u_dim = u_dim # input dimension
        self.y_dim = y_dim # output dimension
        self.hid_dim = hid_dim # hidden state dimension
        self.cell_dim = hid_dim # observable state dimension

        # self.m = self.y_dim # state dimension
        # self.n = self.u_dim # input dimension
        self.rollout_controller = None
        self.target = jax.random.uniform(generate_key(), shape=(self.y_dim,), minval=-1, maxval=1)

        glorot_init = stax.glorot() # returns a function that initializes weights
        self.W_hh = glorot_init(generate_key(), (4*self.hid_dim, self.hid_dim)) # maps h_t to gates
        self.W_uh = glorot_init(generate_key(), (4*self.hid_dim, self.u_dim)) # maps x_t to gates
        self.b_h = np.zeros(4*self.hid_dim)
        self.b_h = jax.ops.index_update(self.b_h, jax.ops.index[self.hid_dim:2*self.hid_dim], np.ones(self.hid_dim)) # forget gate biased initialization
        self.W_out = glorot_init(generate_key(), (self.y_dim, self.hid_dim)) # maps h_t to output
        # self.cell = np.zeros(self.hid_dim) # long-term memory
        # self.hid = np.zeros(self.hid_dim) # short-term memory
        self.hid_cell = np.hstack((np.zeros(self.hid_dim), np.zeros(self.hid_dim)))

        '''
        def _step(x, hid, cell):
            sigmoid = lambda x: 1. / (1. + np.exp(-x)) # no JAX implementation of sigmoid it seems?
            gate = np.dot(self.W_hh, hid) + np.dot(self.W_uh, x) + self.b_h 
            i, f, g, o = np.split(gate, 4) # order: input, forget, cell, output
            next_cell =  sigmoid(f) * cell + sigmoid(i) * np.tanh(g)
            next_hid = sigmoid(o) * np.tanh(next_cell)
            y = np.dot(self.W_out, next_hid)
            return (next_hid, next_cell, y)'''

        def _dynamics(hid_cell_state, u):
            hid = hid_cell_state[:self.hid_dim]
            cell = hid_cell_state[self.hid_dim:]

            sigmoid = lambda u: 1. / (1. + np.exp(-u))
            gate = np.dot(self.W_hh, hid) + np.dot(self.W_uh, u) + self.b_h
            i, f, g, o = np.split(gate, 4) # order: input, forget, cell, output
            next_cell = sigmoid(f) * cell + sigmoid(i) + np.tanh(g)
            next_hid = sigmoid(o) * np.tanh(next_cell)
            y = np.dot(self.W_out, next_hid)
            return (np.hstack((next_hid, next_cell)), y)

        self._dynamics = jax.jit(_dynamics) # MUST store as self._dynamics for default rollout implementation to work
        # C_x, C_u = (np.diag(np.array([0.2, 0.05, 1.0, 0.05])), np.diag(np.array([0.05])))
        # self._loss = jax.jit(lambda x, u: x.T @ C_x @ x + u.T @ C_u @ u) # MUST store as self._loss
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


        # self._step = jax.jit(_step)
        # return np.dot(self.W_out, self.hid)
        return np.dot(self.W_out, self.hid_cell[:self.hid_dim])

    def rollout(self, controller, T, dynamics_grad=False, loss_grad=False, loss_hessian=False):
        # Description: Roll out trajectory of given baby_controller.
        if self.rollout_controller != controller: self.rollout_controller = controller
        x = self._state
        trajectory = self._rollout(controller.get_action, self._dynamics, x, T)
        transcript = {'x': trajectory[:,:self.u_dim], 'u': trajectory[:,self.u_dim:]}

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
            u (numpy.ndarray): RNN input, an n-dimensional real-valued vector.
        Returns:
            The output of the RNN computed on the past l inputs, including the new x.
        """
        assert self.initialized
        assert u.shape == (self.u_dim,)
        self.T += 1

        # self.hid, self.cell, y = self._step(x, self.hid, self.cell)
        self.hid_cell, y = self._dynamics(self.hid_cell, u)
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
        # return (self.hid, self.cell)
        return self.hid_cell


