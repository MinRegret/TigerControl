
"""
Gradient Pertubation Controller

written by Paula Gradu, Elad Hazan and Anirudha Majumdar 
"""

import jax.numpy as np
import tigercontrol
from tigercontrol.methods.control import ControlMethod
from jax import grad,jit
import jax.random as random
from tigercontrol.utils import generate_key
import jax
import scipy

class GPC(ControlMethod):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    compatibles = set([])

    def __init__(self):
        self.initialized = False

    def initialize(self, A, B, x, n, m, H, HH, K = None):
        """
        Description: Initialize the dynamics of the model
        Args:
            A,B (float/numpy.ndarray): system dynamics
            n (float/numpy.ndarray): dimension of the state
            m (float/numpy.ndarray): dimension of the controls
            H (postive int): history of the controller 
            HH history of the system 
            x (float/numpy.ndarray): current state
            past_w (float/numpy.ndarray)  previous perturbations 
        """
        self.initialized = True
        
        def _update_past(self_past, x):
            new_past = np.roll(self_past, self.n)
            new_past = jax.ops.index_update(new_past, 0, x)
            return new_past
        self._update_past = jit(_update_past)

        if(K is None):
            # solve the ricatti equation 
            X = scipy.linalg.solve_continuous_are(A, B, np.identity(n), np.identity(m))
            #compute LQR gain
            self.K = np.linalg.inv(B.T @ X @ B + np.identity(m)) @ (B.T @ X @ A)
        else:
            self.K = K

        self.x = np.zeros(n)        
        self.u = np.zeros(m)
        
        self.n = n   ## dimension of  the state x 
        self.m = m   ## dimension of the control u
        self.A = A
        self.B = B
        self.H = H   ## how many control matrices
        self.HH = HH ## how many times to unfold the recursion

        ## internal parmeters to the class 
        self.T = 1 ## keep track of iterations, for the learning rate
        self.learning_rate = 1
        self.M = np.zeros((H, m, n))
        #self.M = random.normal(generate_key(), shape=(H, m, n)) / np.sqrt(0.5*(n+m)) # Glorot CANNOT BE SET TO ZERO
        self.S = np.repeat(B.reshape(1, n, m), HH, axis=0) # previously [B for i in range(HH)]
        for i in range(1, HH):
            self.S = jax.ops.index_update(self.S, i, (A - B @ self.K) @ self.S[i-1]) 
        self.w_past = np.zeros((HH + H,n)) ## this are the previous perturbations, from most recent [0] to latest [HH-1]

        self.is_online = True

        # This is the counterfactual loss function, we prefer not to differentiate it and use JAX 
        def the_complicated_loss_function(M, w_past):
            final = 0.0
            for i in range(self.HH):
                temp = np.tensordot(M, w_past[i:i+self.H], axes=([0,2],[0,1]))
                final = final + self.S[i] @ temp
            return np.sum(final ** 2)

        self.grad_fn = jit(grad(the_complicated_loss_function))  # compiled gradient evaluation function

        def _plan(x_new, x_old, u_old, w_past, M, lr):
            w_new = x_new - np.dot(self.A, x_old) - np.dot(self.B, u_old)
            w_past_new = self._update_past(w_past, w_new)
            u_new = -self.K @ x_new + np.tensordot(M, w_past[:self.H], axes=([0,2],[0,1]))
            M_new = M - lr * self.grad_fn(M, w_past_new)
            return x_new, u_new, w_past_new, M_new
        self._plan = jit(_plan)

    def plan(self, x_new):
        """
        Description: Updates internal parameters and then returns the estimated optimal action (only one)
        Args:
            None
        Returns:
            Estimated optimal action
        """

        self.T +=1
        lr = self.learning_rate / np.sqrt(self.T)
        self.x, self.u, self.w_past, self.M = self._plan(x_new, self.x, self.u, self.w_past, self.M, lr)
        return self.u


    def help(self):
        """
        Description: Prints information about this class and its methods.
        Args:
            None
        Returns:
            None
        """
        print(GPC_help)

    def __str__(self):
        return "<GPC Model>"

# string to print when calling help() method
GPC_help = """

-------------------- *** --------------------

Id: GPC

Description: Computes regret-minimizing controls using the Gradient Pertubation Controller algorithm.

Methods:

    initialize(A,B, x , n , m , H, HH, K)
        Description:
            Initialize the dynamics of the method
        Args:
            A,B (float/numpy.ndarray): system dynamics
            K  (float/numpy.ndarray): optimal controller 
            n (float/numpy.ndarray): dimension of the state
            m (float/numpy.ndarray): dimension of the controls
            H (postive int): history of the controller 
            HH history of the system 
            x (float/numpy.ndarray): current state
            past_w (float/numpy.ndarray)  previous perturbations 

    step()
        Description: Updates internal parameters and then returns the
        	estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions

    predict()
        Description:
            Returns estimated optimal set of actions
        Args:
            None
        Returns:
            Estimated optimal set of actions

    update()
        Description:
        	Updates internal parameters
        Args:
            None

    help()
        Description:
            Prints information about this class and its methods.
        Args:
            None
        Returns:
            None

-------------------- *** --------------------

"""
