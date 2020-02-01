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

    def initialize(self, A, B, n, m, H, K, delta, x):
        """
        Description: Initialize the dynamics of the model
        Args:
            A,B (float/numpy.ndarray): system dynamics
            K  (float/numpy.ndarray): optimal controller 
            n (float/numpy.ndarray): dimension of the state
            m (float/numpy.ndarray): dimension of the controls
            H (postive int): history of the controller 
        """
        self.initialized = True
        
        def _generate_uniform(shape, norm=1.00):
            v = random.normal(generate_key(), shape=shape)
            v = norm * v / np.linalg.norm(v)
            return v
        self._generate_uniform = _generate_uniform
        self.eps = self._generate_uniform((H, m, n))
        
        self.K = np.zeros ((m,n)) ## compute it...

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
        self.learning_rate = 0.1
        self.M = self._generate_uniform((H, m, n), norm = 1-delta) ## CANNOT BE SET TO ZERO
        self.w_past = np.zeros((H,n)) ## this are the previous perturbations, from most recent [0] to latest [HH-1]
        
    def get_action(self):
        
        M_tilde = self.M + self.delta * self.eps 
        
        #choose action
        self.u = -K @ self.x
        for i in range(self.H):
            self.u += np.dot(M_tilde[i] , self.w_past[i])
            
        return self.u

    def update(self, c_t, x_new):
        """
        Description: Updates internal parameters and then returns the estimated optimal action (only one)
        Args:
            None
        Returns:
            Estimated optimal action
        """

        self.T += 1
        self.learning_rate = 1 / self.T**0.75 # eta_t = O(t^(-3/4))
        
        #get new noise
        w_new = x_new - np.dot(self.A , self.x)  - np.dot(self.B , self.u)
        
        #update past noises
        self.w_past = np.roll(self.w_past, self.n)
        self.w_past  = jax.ops.index_update(self.w_past , 0, w_new)
            
        #set current state
        self.x = x_new
            
        g_t = (self.m * self.n * self.H / self.delta) * c_t * np.sum(self.eps, axis = 0)
        lr = self.learning_rate / self.T**0.75 # eta_t = O(t^(-3/4))
        self.M = (1-self.delta) * (self.M - lr * g_t)
        self.M /= np.linalg.norm(self.M)

        