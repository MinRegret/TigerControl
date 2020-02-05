import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import jax.scipy.linalg as linalg
import numpy as onp
import jax.random as random
import tigercontrol
from tigercontrol.utils.random import set_key, generate_key
from tigercontrol.environments import Environment
from tigercontrol.controllers import Controller
from jax import grad,jit

class SystemID(Controller):
    """
    Description: Identifies unknown matrices A, B of linear dynamical system.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, n, m, K=None, k=None, T_0=None, gamma=0.001):
        """
        Description: Initialize the dynamics of the model
        Args:
            n, m (int): system dynamics dimension
            K (np.array): control matrix
            k (int): history to use when estimating A, B
            T_0 (int): number of total iterations (recommend T_0 = O(T^(2/3)))
            gamma (float): L2 regularization
        """
        self.initialized = True
        self.n, self.m = n, m
        self.T = 0
        self.K = K if K else np.zeros((m, n))
        self.k = k
        self.T_0 = T_0 # our implementation uses T instead of T_0
        self.eta = [] # perturbations
        self.x_history = [] # history of x's
        self.gamma = gamma

    def get_action(self, x_t):
        """ return action """
        self.T += 1
        eta_t = 1 - 2*random.randint(generate_key(), minval=0, maxval=2, shape=(self.m,))
        self.eta.append(eta_t)
        self.x_history.append(np.squeeze(x_t, axis=1))
        return - self.K @ x_t + np.expand_dims(eta_t, axis=1)

    def system_id(self):
        """ returns current estimate of hidden system dynamics """
        assert self.T > 0
        k = self.k if self.k else int(0.15 * self.T)

        # transform eta and x
        eta_np = np.array(self.eta)
        x_np = np.array(self.x_history)

        # prepare vectors and retrieve B
        scan_len = self.T-k-1 # need extra -1 because we iterate over j=0,..,k
        N_j = np.array([np.dot(x_np[j+1:j+1+scan_len].T, eta_np[:scan_len]) for j in range(k+1)]) / scan_len
        B = N_j[0] # np.dot(x_np[1:].T, eta_np[:-1]) / (self.T-1)
        #B = np.dot(x_np[1:].T, eta_np[:-1]) / (self.T-1)
        # retrieve A
        C_0, C_1 = N_j[:-1], N_j[1:]
        C_inv = np.linalg.inv(np.tensordot(C_0, C_0, axes=([0,2],[0,2])) + self.gamma * np.identity(self.n))
        A = np.tensordot(C_1, C_0, axes=([0,2],[0,2])) @ C_inv + B @ self.K

        return (A, B)


if __name__ == "__main__":

    T = 1000
    T_0 = 5000 # system identification
    n, m = 3, 3
    #n, m = 4, 4
    x0 = np.zeros((n, 1))

    # LDS specification
    #A, B = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]), np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    A, B = np.identity(n), np.identity(m)
    #A = random.normal(generate_key(), shape=(n,n))
    #A = A / np.linalg.norm(A, ord='nuc')
    A, B = 0.9 * A, 0.9 * B

    W = np.zeros(T_0 + T)

    sysid = SystemID()
    sysid.initialize(n, m)
    x = x0
    for t in range(T_0):
        u = sysid.get_action(x)
        x = A @ x + B @ u + W[t]
    A_id, B_id = sysid.system_id()
    print("A, B: ", A_id, B_id)
    print("max diff for A, B: ", np.max(np.abs(A - A_id)), np.max(np.abs(B - B_id)))
    print("norm diff for A, B: ", np.linalg.norm(A - A_id), np.linalg.norm(B - B_id))



        