
"""
Gradient Pertubation Controller

written by Paula Gradu, Elad Hazan and Anirudha Majumdar 
"""

import jax.numpy as np
import tigercontrol
from tigercontrol.controllers import Controller
from jax import grad,jit
import jax.random as random
from tigercontrol.utils import generate_key
import jax
import scipy

class GPC(Controller):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    compatibles = set([])

    def __init__(self):
        self.initialized = False

    def _get_dims(self):
        try:
            n = self.B.shape[0] ## dimension of  the state x 
        except:
            n = 1
        try:
            m = self.B.shape[1] ## dimension of the control u
        except:
            m = 1
        return (n, m)

    def initialize(self, A, B, H = 3, HH = 30, K = None, x = None, loss_fn = None):
        """
        Description: Initialize the dynamics of the model
        Args:
            A,B (float/numpy.ndarray): system dynamics
            H (postive int): history of the controller 
            HH (positive int): history of the system 
            K (float/numpy.ndarray): Starting policy (optional). Defaults to LQR gain.
            x (float/numpy.ndarray): initial state (optional)
        """
        self.initialized = True

        self.A, self.B = A, B
        self.n, self.m = self._get_dims()
        self.H, self.HH = H, HH

        self.loss_fn = lambda x, u: np.sum(x**2 + u**2) if loss_fn is None else loss_fn

        def _generate_uniform(shape, norm=1.00):
            v = random.normal(generate_key(), shape=shape)
            v = norm * v / np.linalg.norm(v)
            return v
        self._generate_uniform = _generate_uniform

        if(K is None):
            # solve the ricatti equation 
            X = scipy.linalg.solve_discrete_are(A, B, Q, R)
            #compute LQR gain
            self.K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
        else:
            self.K = K

        self.x = np.zeros(self.n) if x is None else x
        self.u = np.zeros(self.m)

        ## internal parmeters to the class 
        self.T = 1 ## keep track of iterations, for the learning rate
        self.learning_rate = 0.1
        self.M = self._generate_uniform((H, self.m, self.n), norm = 0.1)
        self.S = np.repeat(B.reshape(1, self.n, self.m), HH, axis=0) # previously [B for i in range(HH)]
        for i in range(1, HH):
            self.S = jax.ops.index_update(self.S, i, (A - B @ self.K) @ self.S[i-1]) 
        self.w_past = np.zeros((HH + H, self.n)) ## this are the previous perturbations, from most recent [0] to latest [HH-1]

        # This is the counterfactual loss function, we prefer not to differentiate it and use JAX 
        ''' def the_complicated_loss_function(M, w_past):
            x = 0.0
            for i in range(self.HH):
                psi = 0.0

                for j in range(i, i + self.H):
                    psi += self.S[i] @ M[i - j]
                x += psi @ w_past[i]

            u = - K @ x
            for i in range(self.H):
                u += np.dot(M[-i-1] , w_past[i])

            return loss_fn(x, u) '''

        #self.grad_fn = jit(grad(the_complicated_loss_function))  # compiled gradient evaluation function

        # new attept at defining counterfact loss fn
        def counterfact_loss(M, w):
            y, cost = np.zeros(self.n), 0
            for h in range(HH - H - 1):
                v = -self.K @ y + np.tensordot(M, w[h : h + H], axes = ([0, 2], [0, 1]))
                y = A @ y + B @ v + w[h + H]
            v = -self.K @ y + np.tensordot(M, w[h : h + H], axes = ([0, 2], [0, 1])) 
            cost = loss_fn(y, v)
            return cost

        self.grad_fn = grad(counterfact_loss)
        

    def get_action(self):
        """
        Description: Return the action based on current state and internal parameters.

        Args:
            x (float/numpy.ndarray): current state

        Returns:
            u(float/numpy.ndarray): action to take
        """

        
        self.u = -self.K @ self.x
        for i in range(self.H):
            self.u += np.dot(self.M[-i-1] , self.w_past[i])

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
        lr = self.learning_rate / np.sqrt(self.T)
        
        #get new noise
        w_new = x_new - np.dot(self.A , self.x)  - np.dot(self.B , self.u)
        
        #update past noises
        self.w_past = np.roll(self.w_past, self.n)
        self.w_past  = jax.ops.index_update(self.w_past , 0, w_new)
            
        #set current state
        self.x = x_new

        self.M = self.M - lr * self.grad_fn(self.M, self.w_past)
        self.M /= np.linalg.norm(self.M)

    def __str__(self):
        return "<GPC Model>"

