
import jax
import jax.numpy as np
import jax.scipy.linalg as linalg
#import jax.random as random
import numpy.random as random
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
        self.initialized = True
        self.n, self.m = n, m
        self.T = 0
        self.K = K if K else np.zeros((m, n))
        self.lr = learning_rate
        self.stash = []
        self.x_history = []
        self.u_history = []

        # initialize matrices
        self.A = np.identity(n)
        self.B = np.zeros((n, m))

    def get_action(self, x_t, done):
        """ return action """
        self.T += 1
        # regular numpy
        eta_t = 1 - 2*random.randint(2, size=(self.m,)) 
        u_t = - self.K @ x_t + np.expand_dims(eta_t, axis=1)
        self.x_history.append(np.squeeze(x_t, axis=1))
        self.u_history.append(np.squeeze(u_t, axis=1))
        #u_t = - self.K @ x_t + eta_t
        #self.x_history.append(x_t)
        #self.u_history.append(u_t)
        if done:
          if len(self.x_history) > 1:
            self.stash.append((self.x_history, self.u_history))
          self.x_history = []
          self.u_history = []
        return u_t

    def system_id(self):
        """ returns current estimate of hidden system dynamics """
        assert self.T > 1 # need at least 2 data points
        if len(self.x_history) > 0:
          self.stash.append((self.x_history, self.u_history))

        # transform x and u into regular numpy arrays for least squares
        x_t = onp.vstack([onp.array(x[:-1]) for x, u in self.stash])
        u_t = onp.vstack([onp.array(u[:-1]) for x, u in self.stash])
        x_t1 = onp.vstack([onp.array(x[1:]) for x, u in self.stash])

        # regression on A and B jointly
        A_B = scilinalg.lstsq(np.hstack((x_t, u_t)), x_t1)[0]
        A, B = np.array(A_B[:self.n]).T, np.array(A_B[self.n:]).T
        return (A, B)


if __name__ == "__main__":

    T = 1000
    T_0 = 1000 # system identification
    T_split = 100
    n, m = 3, 3
    #n, m = 4, 4
    x0 = np.zeros((n, 1))
    #x0 = np.zeros((n,))

    def random_matrix(shape):
        m = random.normal(size=shape)
        return m / np.linalg.norm(m, ord='nuc')

    # LDS specification
    #A, B = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]), np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    A, B = np.identity(n), np.identity(m)
    A, B = random_matrix((n,n)), random_matrix((n,m))
    A, B = 0.9 * A, 0.9 * B

    W = np.zeros(T_0 + T)

    sysid = RegressionSystemID()
    sysid.initialize(n, m)
    x = x0
    for t in range(T_0):
        done = ((t+1) % T_split == 0)
        u = sysid.get_action(x, done)
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



        