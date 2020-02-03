import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import numpy as onp
import jax.random as random
import tigercontrol
from tigercontrol.utils.random import set_key, generate_key
from tigercontrol.environments import Environment
from tigercontrol.controllers import Controller
from jax import grad,jit

class BPC(Controller):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """

    def __init__(self):
        self.initialized = False

    def initialize(self, A, B, n, m, H, K, delta, x, initial_lr=0.1):
        """
        Description: Initialize the dynamics of the model
        Args:
            A,B (float/numpy.ndarray): system dynamics
            n (float/numpy.ndarray): dimension of the state
            m (float/numpy.ndarray): dimension of the controls
            H (postive int): history of the controller 
            K (float/numpy.ndarray): optimal controller 
            delta (float): gradient estimator parameter
            x (numpy.ndarray): initial state
            initial_lr (float): initial learning rate
        """
        self.initialized = True
        
        def _generate_uniform(shape, norm=1.00):
            v = random.normal(generate_key(), shape=shape)
            v = norm * v / np.linalg.norm(v)
            return v
        self._generate_uniform = _generate_uniform
        self.eps = self._generate_uniform((H, H, m, n))
        
        self.K = np.zeros((m, n)) ## compute it...

        self.x = x        
        self.u = np.zeros(m)
        
        self.n = n   ## dimension of  the state x 
        self.m = m   ## dimension of the control u
        self.A = A
        self.B = B
        self.H = H   ## how many control matrices
    
        self.delta = delta

        ## internal parmeters to the class 
        self.T = 1 ## keep track of iterations, for the learning rate
        self.learning_rate = initial_lr
        self.M = self._generate_uniform((H, m, n), norm = 1-delta) ## CANNOT BE SET TO ZERO
        self.w_past = np.zeros((H,n)) ## this are the previous perturbations, from most recent [0] to latest [HH-1]
        
    def get_action(self):
        M_tilde = self.M + self.delta * self.eps[-1]
        print("M_tilde ", M_tilde)
        print("w_past ", self.w_past)
        #choose action
        self.u = -self.K @ self.x + np.tensordot(M_tilde, self.w_past, axes=([0, 2], [0, 1]))
        return self.u

    def update(self, c_t, x_new):
        """
        Description: Updates internal parameters
        Args:
            c_t (float): loss at time t
            x_new (array): next state
        Returns:
            Estimated optimal action
        """

        self.T += 1
        #self.learning_rate = 1 / self.T**0.75 # eta_t = O(t^(-3/4)) # bug?

        # update noise
        next_norm = np.sqrt(1 - np.sum(self.eps[:-1] **2))
        next_eps = self._generate_uniform((self.H, self.m, self.n), norm=next_norm)
        self.eps = np.roll(self.eps, -(self.H * self.m * self.n))
        self.eps = jax.ops.index_update(self.eps, -1, next_eps)

        #get new noise
        w_new = x_new - np.dot(self.A , self.x)  - np.dot(self.B , self.u)
        
        #update past noises
        self.w_past = np.roll(self.w_past, -self.n)
        self.w_past = jax.ops.index_update(self.w_past, -1, w_new)
            
        #set current state
        self.x = x_new
            
        # gradient estimate and update
        g_t = (self.m * self.n * self.H / self.delta) * c_t * np.sum(self.eps, axis = 0)
        lr = self.learning_rate / self.T**0.75 # eta_t = O(t^(-3/4))
        self.M = (self.M - lr * g_t)
        curr_norm = np.linalg.norm(self.M)
        if curr_norm > (1 - self.delta):
            self.M *= (1-self.delta) / curr_norm
            
        return self.u

        