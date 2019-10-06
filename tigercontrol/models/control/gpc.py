
"""
Gradient Pertubation Controller

written by Paula Gradu, Elad Hazan and Anirudha Majumdar 
"""

import jax.numpy as np
import tigercontrol
from tigercontrol.models.control import ControlModel
from jax import grad,jit

class GPC(ControlModel):
    """
    Description: Computes optimal set of actions using the Linear Quadratic Regulator
    algorithm.
    """
    
    compatibles = set([])

    def __init__(self):
        self.initialized = False

    def to_ndarray(self, x):
        """
        Description: If x is a scalar, transform it to a (1, 1) numpy.ndarray;
        otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
        Returns:
            A numpy.ndarray representation of x
        """
        x = np.asarray(x)
        if(np.ndim(x) == 0):
            x = x[None, None]
        return x

    def extend(self, x, T):
        """
        Description: If x is not in the correct form, convert it; otherwise, leave it unchanged.
        Args:
            x (float/numpy.ndarray)
            T (postive int): number of timesteps
        Returns:
            A numpy.ndarray representation of x
        """
        x = self.to_ndarray(x)
        return [x for i in range(T)]

    def initialize(self, A,B, x , n , m , H, HH, K  ):
        """
        Description: Initialize the dynamics of the model
        Args:
            A,B (float/numpy.ndarray): system dynamics
            K  (float/numpy.ndarray): optimal controller 
            n (float/numpy.ndarray): dimension of the state
            m (float/numpy.ndarray): dimension of the controls
            H (postive int): history of the controller 
            HH history of the system 
            x (float/numpy.ndarray): current state
            past_w (float/numpy.ndarray)  previous perturbations 
        """
        self.initialized = True
        
        def _update_past(self_past, x):
            new_past = np.roll(self_past, 1)
            new_past = jax.ops.index_update(new_past, 0, x)
            return new_past
        self._update_past = jax.jit(_update_past)
        
        self.K = np.zeros ((n,n)) ## compute it...

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
        self.M = np.zeros((m, n,H) )
        self.S = np.zeros( (n,m,HH )  )
        S[0] = B
        for i in range(HH)
            S[i] = (A + B@K) @ S[i-1]
        self.past_w = np.zeros((HH+H,n)) ## this are the previous perturbations, from most recent [0] to latest [HH-1]



    def the_complicated_loss_function(self):
        """
        This is the counterfactual loss function, we prefer not to differentiate it and use JAX 
        """
        final = np.zeros(n)
        for i in range(self.HH):
            temp = np.zeros(m)
            for j in range(self.H)
                temp = temp + np.dot( M[j] , self.past_w[i+j])
            final = final + S[i] @ temp
        return np.norm(final)




    def plan(self,x_new):
        """
        Description: Updates internal parameters and then returns the estimated optimal action (only one)
        Args:
            None
        Returns:
            Estimated optimal action
        """

        self.T +=1
        self.learning_rate = 1 / np.sqrt(self.T + 1)

        w_new = x_new - np.dot(self.A , self.x)  - np.dot(self.B , self.u)
        self._update_past(self.w, w_new)
        self.x = x_new

        self.u = np.zeros(self.m)
        for i in range(self.H):
            self.u += np.dot( self.M[i] , self.w[i] )

        M = M - self.learning_rate * grad(jit(the_complicated_loss_function))
        return self.u


        ## update the M matrices via OGD rule ##



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
        return "<GC Model>"


# string to print when calling help() method
GPC_help = """

-------------------- *** --------------------

Id: GPC

Description: Computes regret-minimizing controls using the Gradient Pertubation Controller algorithm.

Methods:

    initialize(A,B, x , n , m , H, HH, K)
        Description:
            Initialize the dynamics of the model
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