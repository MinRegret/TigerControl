
import jax
import jax.numpy as np
import jax.scipy.linalg as linalg
import jax.random as random
from jax import grad,jit

import numpy as onp
from scipy import linalg as scilinalg # needed for least squares
import tigercontrol
import matplotlib.pyplot as plt

from tigercontrol.utils.random import set_key, generate_key
from tigercontrol.environments import Environment
from tigercontrol.controllers import Controller


class RegressionSystemID(Controller):
    """
    Description: Identifies unknown matrices A, B of linear dynamical system.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, K=None, learning_rate=0.001):
        """
        Description: Initialize the dynamics of the model
        Args:
            n, m (int): system dynamics dimension
            K (np.array): control matrix
        """
        self.initialized = True
        self.n, self.m = n, m
        self.T = 0
        self.K = K if K else np.zeros((m, n))
        self.lr = learning_rate
        self.x_history = []
        self.u_history = []

        # initialize matrices
        self.A = np.identity(n)
        self.B = np.zeros((n, m))

    def get_action(self, x_t):
        """ return action """
        self.T += 1
        eta_t = 1 - 2*random.randint(generate_key(), minval=0, maxval=2, shape=(self.m,))
        u_t = - self.K @ x_t + np.expand_dims(eta_t, axis=1)
        self.x_history.append(np.squeeze(x_t, axis=1))
        self.u_history.append(np.squeeze(u_t, axis=1))
        return u_t

    def system_id(self):
        """ returns current estimate of hidden system dynamics """
        assert self.T > 1 # need at least 2 data points

        # transform x and u into regular numpy arrays for least squares
        x_np = onp.array(self.x_history)
        u_np = onp.array(self.u_history)

        # regression on A and B jointly
        A_B = scilinalg.lstsq(np.hstack((x_np[:-1], u_np[:-1])), x_np[1:])[0]
        A, B = np.array(A_B[:self.n]).T, np.array(A_B[self.n:]).T
        return (A, B)


if __name__ == "__main__":

    T = 1000
    T_0 = 5000 # system identification
    n, m = 5, 3
    #n, m = 4, 4
    x0 = np.zeros((n, 1))

    def random_matrix(shape):
        m = random.normal(generate_key(), shape=shape)
        return m / np.linalg.norm(m, ord='nuc')

    # LDS specification
    #A, B = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]), np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    #A, B = np.identity(n), np.identity(m)
    #A = random_matrix((n,n))
    #B = random_matrix((n,m))
    #A, B = 0.9 * A, 0.9 * B

    A = np.array([[9., 1., 0., 0., 0.],
              [0., 9., 0., 0., 0.],
              [0., 0., 5., 1., 0.],
              [0., 0., 0., 5., 0.],
              [0., 0., 0., 0., 1.]])

    A = np.identity(5)

    B = np.array([[1., 0., 0.],
              [0.5, 0.5, 0.],
              [0.5, 0.0, 0.5],
              [0., 1., 0.],
              [0., 0.5, 0.5]])

    A = 0.9 * A / np.linalg.norm(A, ord='nuc')
    B = 0.9 * B / np.linalg.norm(B, ord='nuc')


    #W = np.zeros(T_0 + T)
    W = (np.sin(np.arange((T_0+T)*m)/(20*np.pi)).reshape((T_0+T),m) @ np.ones((m, n))).reshape((T_0+T), n, 1)

    sysid = RegressionSystemID()
    sysid.initialize(n, m)
    x = x0
    for t in range(T_0):
        u = sysid.get_action(x)
        x = A @ x + B @ u + W[t]
    A_id, B_id = sysid.system_id()
    print("A versus A_id")
    print(A)
    print(A_id)
    print("B versus B_id")
    print(B)
    print(B_id)
    print("max diff for A, B: ", np.max(np.abs(A - A_id)), np.max(np.abs(B - B_id)))
    print("norm diff for A, B: ", np.linalg.norm(A - A_id), np.linalg.norm(B - B_id))



        